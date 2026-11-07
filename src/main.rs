use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use h3o::{CellIndex, LatLng, Resolution};
use osmpbf::{Element, ElementReader};
use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::Deserialize;

use h3_transit::{
    canonical_pair, resolve_lookup_pair, DistanceTier, H3CsrIndex, ParametricModel, TransitSummary,
    TIER1_RES8_MAX_RINGS, TIER2_RES7_MAX_RINGS, TIER3_RES6_MAX_RINGS, TIER4_RES5_MAX_RINGS,
};

#[derive(Parser, Debug)]
#[command(
    name = "h3-transit",
    about = "Multi-resolution H3 stochastic road-network travel time builder & PIT backtester"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Distills an .osm.pbf + running osrm-routed instance into a packed CSR binary index (.h3t).
    Build {
        /// Path to the regional .osm.pbf file (e.g., osrm-data/nj-buffered.osm.pbf)
        #[arg(long)]
        pbf: PathBuf,
        /// Output path for the packed CSR binary index (e.g., osrm-data/nj_h3_transit.h3t)
        #[arg(long)]
        db: PathBuf,
        /// URL of the local osrm-routed server
        #[arg(long, default_value = "http://127.0.0.1:5050")]
        osrm_url: String,
        /// Core bounding box `min_lon,min_lat,max_lon,max_lat` for origin hexes
        #[arg(long, default_value = "-75.6,38.9,-73.85,41.36")]
        core_bbox: String,
        /// Number of spatially stratified road points per Res 8 hex for inter-hex routing
        #[arg(long, default_value_t = 3)]
        samples_per_res8: usize,
    },
    /// Runs a Probability Integral Transform (PIT) holdout backtest comparing distribution families.
    Backtest {
        /// Path to the regional .osm.pbf file
        #[arg(long)]
        pbf: PathBuf,
        /// Path to the built packed CSR binary index (.h3t)
        #[arg(long)]
        db: PathBuf,
        /// URL of the local osrm-routed server for ground-truth point-to-point routes
        #[arg(long, default_value = "http://127.0.0.1:5050")]
        osrm_url: String,
        /// Core bounding box `min_lon,min_lat,max_lon,max_lat`
        #[arg(long, default_value = "-75.6,38.9,-73.85,41.36")]
        core_bbox: String,
        /// Number of holdout random road-node pairs per distance tier
        #[arg(long, default_value_t = 2000)]
        pairs_per_tier: usize,
    },
    /// Queries the travel time distribution between two H3 Res-8 hexes.
    Query {
        /// Path to the packed CSR binary index (.h3t)
        #[arg(long)]
        db: PathBuf,
        /// Origin H3 cell index (hex string, e.g. 882a100d29fffff)
        #[arg(long)]
        orig: String,
        /// Destination H3 cell index (hex string)
        #[arg(long)]
        dest: String,
    },
    /// Merges multiple regional .h3t CSR index files into a single unified .h3t file.
    Merge {
        /// Output path for the unified .h3t file
        #[arg(long)]
        output: PathBuf,
        /// Input regional .h3t files
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
    },
    /// Iteratively tiles the entire United States (.osm.pbf), recursively bisecting dense
    /// regions so osrm-extract stays under RAM limits, and merges all tiles into a unified .h3t index.
    BuildUs {
        /// Path to the full US .osm.pbf file
        #[arg(long, default_value = "osrm-data/us-261103.osm.pbf")]
        us_pbf: PathBuf,
        /// Working directory for temporary PBF splits and completed tile .h3t checkpoints
        #[arg(long, default_value = "osrm-data/us_build")]
        work_dir: PathBuf,
        /// Final merged output path
        #[arg(long, default_value = "osrm-data/us_h3_transit.h3t")]
        output: PathBuf,
        /// Maximum buffered PBF size (in MB) before recursively bisecting a region in half
        #[arg(long, default_value_t = 450)]
        max_tile_pbf_mb: u64,
        /// Port for the managed local osrm-routed instance
        #[arg(long, default_value_t = 5050)]
        osrm_port: u16,
    },
}

#[derive(Clone, Copy, Debug)]
struct BBox {
    min_lon: f64,
    min_lat: f64,
    max_lon: f64,
    max_lat: f64,
}

impl BBox {
    fn parse(s: &str) -> Result<Self> {
        let parts: Vec<f64> = s
            .split(',')
            .map(|p| p.trim().parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        anyhow::ensure!(parts.len() == 4, "Expected min_lon,min_lat,max_lon,max_lat");
        Ok(Self {
            min_lon: parts[0],
            min_lat: parts[1],
            max_lon: parts[2],
            max_lat: parts[3],
        })
    }

    #[inline]
    fn contains(&self, lat: f64, lon: f64) -> bool {
        lat >= self.min_lat && lat <= self.max_lat && lon >= self.min_lon && lon <= self.max_lon
    }

    fn to_arg_string(self) -> String {
        format!(
            "{:.4},{:.4},{:.4},{:.4}",
            self.min_lon, self.min_lat, self.max_lon, self.max_lat
        )
    }

    fn padded(self, deg: f64) -> Self {
        Self {
            min_lon: (self.min_lon - deg).max(-179.99),
            min_lat: (self.min_lat - deg).max(-85.0),
            max_lon: (self.max_lon + deg).min(179.99),
            max_lat: (self.max_lat + deg).min(85.0),
        }
    }

    /// Splits the bounding box in half along its wider angular dimension.
    fn split_half(self) -> (Self, Self) {
        let d_lon = (self.max_lon - self.min_lon) * (self.min_lat.to_radians().cos().max(0.4));
        let d_lat = self.max_lat - self.min_lat;
        if d_lon >= d_lat {
            let mid_lon = 0.5 * (self.min_lon + self.max_lon);
            (
                Self {
                    max_lon: mid_lon,
                    ..self
                },
                Self {
                    min_lon: mid_lon,
                    ..self
                },
            )
        } else {
            let mid_lat = 0.5 * (self.min_lat + self.max_lat);
            (
                Self {
                    max_lat: mid_lat,
                    ..self
                },
                Self {
                    min_lat: mid_lat,
                    ..self
                },
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RoadPoint {
    lat: f32,
    lon: f32,
    h10: CellIndex,
}

#[derive(Deserialize)]
struct OsrmTableResponse {
    code: String,
    durations: Option<Vec<Vec<Option<f32>>>>,
}

#[derive(Deserialize)]
struct OsrmRouteLeg {
    duration: f32,
}

#[derive(Deserialize)]
struct OsrmRouteResponse {
    code: String,
    routes: Option<Vec<OsrmRouteLeg>>,
}

fn query_osrm_single_route(
    client: &Client,
    osrm_url: &str,
    orig: RoadPoint,
    dest: RoadPoint,
) -> Option<f32> {
    let url = format!(
        "{}/route/v1/driving/{:.5},{:.5};{:.5},{:.5}?overview=false",
        osrm_url.trim_end_matches('/'),
        orig.lon,
        orig.lat,
        dest.lon,
        dest.lat
    );
    let resp: OsrmRouteResponse = client.get(&url).send().ok()?.json().ok()?;
    if resp.code != "Ok" {
        return None;
    }
    resp.routes?.into_iter().next().map(|r| r.duration)
}

fn is_drivable_highway(val: &str) -> bool {
    matches!(
        val,
        "motorway"
            | "trunk"
            | "primary"
            | "secondary"
            | "tertiary"
            | "unclassified"
            | "residential"
            | "motorway_link"
            | "trunk_link"
            | "primary_link"
            | "secondary_link"
            | "tertiary_link"
            | "living_street"
    )
}

fn extract_road_points_by_res8(pbf_path: &PathBuf) -> Result<HashMap<CellIndex, Vec<RoadPoint>>> {
    let t0 = Instant::now();
    println!(
        "[osm] Pass 1: scanning drivable highway ways in {:?}...",
        pbf_path
    );
    let reader = ElementReader::from_path(pbf_path)?;
    let drivable_nodes: HashSet<i64> = reader.par_map_reduce(
        |element| {
            let mut local = HashSet::new();
            if let Element::Way(way) = element {
                let mut drivable = false;
                for (k, v) in way.tags() {
                    if k == "highway" && is_drivable_highway(v) {
                        drivable = true;
                        break;
                    }
                }
                if drivable {
                    let refs = way.raw_refs();
                    if !refs.is_empty() {
                        let mut id = 0i64;
                        for (idx, delta) in refs.iter().enumerate() {
                            id += *delta;
                            if idx == 0 || idx == refs.len() / 2 || idx + 1 == refs.len() {
                                local.insert(id);
                            }
                        }
                    }
                }
            }
            local
        },
        HashSet::new,
        |mut a, mut b| {
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            a.extend(b);
            a
        },
    )?;
    println!(
        "[osm] Pass 1 complete in {:.1}s: {} unique road vertices.",
        t0.elapsed().as_secs_f64(),
        drivable_nodes.len()
    );

    let t1 = Instant::now();
    println!("[osm] Pass 2: mapping road vertices to H3 Res 8 & Res 10 cells...");
    let reader2 = ElementReader::from_path(pbf_path)?;
    let by_h8: HashMap<CellIndex, Vec<RoadPoint>> = reader2.par_map_reduce(
        |element| {
            let mut local: HashMap<CellIndex, Vec<RoadPoint>> = HashMap::new();
            match element {
                Element::DenseNode(node) => {
                    if drivable_nodes.contains(&node.id) {
                        let lat = node.lat();
                        let lon = node.lon();
                        if let Ok(ll) = LatLng::new(lat, lon) {
                            let h8 = ll.to_cell(Resolution::Eight);
                            let h10 = ll.to_cell(Resolution::Ten);
                            let entry = local.entry(h8).or_default();
                            if entry.len() < 32 {
                                entry.push(RoadPoint {
                                    lat: lat as f32,
                                    lon: lon as f32,
                                    h10,
                                });
                            }
                        }
                    }
                }
                Element::Node(node) => {
                    if drivable_nodes.contains(&node.id()) {
                        let lat = node.lat();
                        let lon = node.lon();
                        if let Ok(ll) = LatLng::new(lat, lon) {
                            let h8 = ll.to_cell(Resolution::Eight);
                            let h10 = ll.to_cell(Resolution::Ten);
                            let entry = local.entry(h8).or_default();
                            if entry.len() < 32 {
                                entry.push(RoadPoint {
                                    lat: lat as f32,
                                    lon: lon as f32,
                                    h10,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
            local
        },
        HashMap::new,
        |mut a, b| {
            for (k, mut v) in b {
                let entry = a.entry(k).or_default();
                if entry.len() < 32 {
                    let take = (32 - entry.len()).min(v.len());
                    entry.extend(v.drain(..take));
                }
            }
            a
        },
    )?;

    println!(
        "[osm] Pass 2 complete in {:.1}s: {} road-active H3 Res 8 cells.",
        t1.elapsed().as_secs_f64(),
        by_h8.len()
    );
    Ok(by_h8)
}

/// Picks up to `k` spatially stratified road points inside a cell by preferring distinct Res 10 sub-cells.
fn select_stratified_points(points: &[RoadPoint], k: usize) -> Vec<RoadPoint> {
    if points.len() <= k {
        return points.to_vec();
    }
    let mut seen_h10 = HashSet::new();
    let mut chosen = Vec::with_capacity(k);
    let step = (points.len() / k).max(1);
    for idx in (0..points.len()).step_by(step) {
        let p = points[idx];
        if seen_h10.insert(p.h10) {
            chosen.push(p);
            if chosen.len() == k {
                return chosen;
            }
        }
    }
    for &p in points {
        if chosen.len() == k {
            break;
        }
        chosen.push(p);
    }
    chosen
}

/// Picks up to `k` unstratified road points across the slice (including nearby/same-neighborhood points)
/// so Tier 0 intra-hex sampling reflects the true distribution of random point pairs inside a single hex.
fn select_unstratified_points(points: &[RoadPoint], k: usize) -> Vec<RoadPoint> {
    if points.len() <= k {
        return points.to_vec();
    }
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let idx = (i * (points.len() - 1)) / (k - 1).max(1);
        out.push(points[idx]);
    }
    out
}

/// Queries OSRM `/table/v1/driving` for `sources` and `destinations`, chunking safely so HTTP URLs
/// never exceed Boost.Beast's header limit.
fn query_osrm_table(
    client: &Client,
    osrm_url: &str,
    sources: &[RoadPoint],
    destinations: &[RoadPoint],
) -> Result<Vec<Vec<Option<f32>>>> {
    if sources.is_empty() || destinations.is_empty() {
        return Ok(Vec::new());
    }

    let mut full_matrix = vec![vec![None; destinations.len()]; sources.len()];
    const MAX_SRC_CHUNK: usize = 500;
    const MAX_DST_CHUNK: usize = 1400;

    for (s_offset, src_chunk) in sources.chunks(MAX_SRC_CHUNK).enumerate() {
        let s_base = s_offset * MAX_SRC_CHUNK;
        for (d_offset, dst_chunk) in destinations.chunks(MAX_DST_CHUNK).enumerate() {
            let d_base = d_offset * MAX_DST_CHUNK;

            let mut coords = String::with_capacity((src_chunk.len() + dst_chunk.len()) * 22);
            for (i, p) in src_chunk.iter().chain(dst_chunk.iter()).enumerate() {
                if i > 0 {
                    coords.push(';');
                }
                coords.push_str(&format!("{:.5},{:.5}", p.lon, p.lat));
            }
            let src_indices: String = (0..src_chunk.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(";");
            let dst_indices: String = (src_chunk.len()..(src_chunk.len() + dst_chunk.len()))
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(";");

            let url = format!(
                "{}/table/v1/driving/{}?sources={}&destinations={}&annotations=duration",
                osrm_url.trim_end_matches('/'),
                coords,
                src_indices,
                dst_indices
            );

            let resp: OsrmTableResponse = client
                .get(&url)
                .send()
                .context("OSRM HTTP request failed")?
                .json()
                .context("Failed to parse OSRM JSON response")?;

            anyhow::ensure!(resp.code == "Ok", "OSRM returned non-Ok code: {}", resp.code);
            let sub = resp.durations.context("Missing durations in OSRM response")?;
            for (si, row) in sub.into_iter().enumerate() {
                for (di, val) in row.into_iter().enumerate() {
                    full_matrix[s_base + si][d_base + di] = val;
                }
            }
        }
    }

    Ok(full_matrix)
}

fn run_build(
    pbf: PathBuf,
    db_path: PathBuf,
    osrm_url: String,
    core_bbox_str: String,
    samples_per_res8: usize,
) -> Result<()> {
    let bbox = BBox::parse(&core_bbox_str)?;
    let by_h8 = extract_road_points_by_res8(&pbf)?;

    let mut rep_by_h8: HashMap<CellIndex, Vec<RoadPoint>> = HashMap::with_capacity(by_h8.len());
    let mut core_h8_cells: Vec<CellIndex> = Vec::new();

    for (&h8, pts) in &by_h8 {
        let reps = select_stratified_points(pts, samples_per_res8);
        let ll = LatLng::from(h8);
        if bbox.contains(ll.lat(), ll.lng()) {
            core_h8_cells.push(h8);
        }
        rep_by_h8.insert(h8, reps);
    }
    core_h8_cells.sort();
    if core_h8_cells.is_empty() {
        println!("[build] Zero road-active cells inside core bbox — writing empty CSR index.");
        H3CsrIndex::write_to_path(&db_path, &[])?;
        return Ok(());
    }

    // Build parent cell lookup maps for Res 7, Res 6, Res 5
    let mut rep_by_h7: HashMap<CellIndex, Vec<RoadPoint>> = HashMap::new();
    let mut rep_by_h6: HashMap<CellIndex, Vec<RoadPoint>> = HashMap::new();
    let mut rep_by_h5: HashMap<CellIndex, Vec<RoadPoint>> = HashMap::new();

    for (&h8, reps) in &rep_by_h8 {
        if let Some(p0) = reps.first().copied() {
            if let Some(h7) = h8.parent(Resolution::Seven) {
                rep_by_h7.entry(h7).or_default().push(p0);
            }
            if let Some(h6) = h8.parent(Resolution::Six) {
                let e = rep_by_h6.entry(h6).or_default();
                if e.len() < 8 {
                    e.push(p0);
                }
            }
            if let Some(h5) = h8.parent(Resolution::Five) {
                let e = rep_by_h5.entry(h5).or_default();
                if e.len() < 8 {
                    e.push(p0);
                }
            }
        }
    }

    println!(
        "[build] Active cells in core bbox: {} Res8 | Total buffered active: {} Res8, {} Res7, {} Res6, {} Res5",
        core_h8_cells.len(),
        rep_by_h8.len(),
        rep_by_h7.len(),
        rep_by_h6.len(),
        rep_by_h5.len()
    );

    // Step 1: Compute Tier 0 (Intra-Hex d8=0, using 6 unstratified points per hex within the Res-6 group)
    // AND Tier 1 (Local Res8, 1 <= d8 <= 8) using canonical half-matrix symmetry so we don't recompute
    // core-to-core pairs in both directions!
    let core_h8_set: HashSet<CellIndex> = core_h8_cells.iter().copied().collect();
    let mut core_by_h6: BTreeMap<CellIndex, Vec<CellIndex>> = BTreeMap::new();
    for &h8 in &core_h8_cells {
        if let Some(h6) = h8.parent(Resolution::Six) {
            core_by_h6.entry(h6).or_default().push(h8);
        }
    }
    let h6_groups: Vec<(CellIndex, Vec<CellIndex>)> = core_by_h6.into_iter().collect();

    let t_t1 = Instant::now();
    println!(
        "[build] Computing Tier 0 (Intra-Res8) & Tier 1 (Res8 Ring 1..8) across {} Res-6 batches...",
        h6_groups.len()
    );
    let completed_batches = std::sync::atomic::AtomicUsize::new(0);
    let total_batches = h6_groups.len();

    let tier01_raw_samples: Vec<((CellIndex, CellIndex), Vec<f32>)> = h6_groups
        .par_iter()
        .map_init(
            || {
                Client::builder()
                    .timeout(std::time::Duration::from_secs(120))
                    .build()
                    .unwrap()
            },
            |client, &(orig_h6, ref orig_h8_list)| {
                let mut local_pairs = Vec::new();

                // 1A. Fast Intra-Hex (Tier 0, d8 = 0) query: only points inside this Res-6 cell (~150x150)
                let mut intra_pts: Vec<RoadPoint> = Vec::with_capacity(orig_h8_list.len() * 6);
                let mut intra_ranges: Vec<(CellIndex, std::ops::Range<usize>)> =
                    Vec::with_capacity(orig_h8_list.len());
                for &orig_h8 in orig_h8_list {
                    if let Some(pts) = by_h8.get(&orig_h8) {
                        let chosen = select_unstratified_points(pts, 6);
                        let start = intra_pts.len();
                        intra_pts.extend_from_slice(&chosen);
                        intra_ranges.push((orig_h8, start..intra_pts.len()));
                    }
                }
                if let Ok(intra_mat) =
                    query_osrm_table(client, &osrm_url, &intra_pts, &intra_pts)
                {
                    for (orig_h8, range) in intra_ranges {
                        let mut samples = Vec::with_capacity(30);
                        for si in range.clone() {
                            if let Some(row) = intra_mat.get(si) {
                                for di in range.clone() {
                                    if si != di {
                                        if let Some(Some(dur)) = row.get(di) {
                                            if *dur > 1.0 && dur.is_finite() {
                                                samples.push(*dur);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if !samples.is_empty() {
                            local_pairs.push(((orig_h8, orig_h8), samples));
                        }
                    }
                }

                // 1B. Half-Matrix Symmetric Tier 1 (1 <= d8 <= 8):
                // If `ring_cell` is also in `core_h8_set`, only query it when `ring_cell.parent(Res6) >= orig_h6`
                // so we cut destination points in half instead of querying A->B and B->A in two separate batches!
                let mut dest_h8_set: HashSet<CellIndex> = HashSet::new();
                for &orig_h8 in orig_h8_list {
                    for ring_cell in orig_h8.grid_disk::<Vec<_>>(TIER1_RES8_MAX_RINGS as u32) {
                        if ring_cell != orig_h8 && rep_by_h8.contains_key(&ring_cell) {
                            let include = if core_h8_set.contains(&ring_cell) {
                                ring_cell
                                    .parent(Resolution::Six)
                                    .map(|dh6| u64::from(dh6) >= u64::from(orig_h6))
                                    .unwrap_or(true)
                            } else {
                                true
                            };
                            if include {
                                dest_h8_set.insert(ring_cell);
                            }
                        }
                    }
                }

                let mut src_points: Vec<RoadPoint> = Vec::new();
                let mut src_ranges: Vec<(CellIndex, std::ops::Range<usize>)> =
                    Vec::with_capacity(orig_h8_list.len());
                for &orig_h8 in orig_h8_list {
                    if let Some(pts) = rep_by_h8.get(&orig_h8) {
                        let start = src_points.len();
                        src_points.extend_from_slice(pts);
                        src_ranges.push((orig_h8, start..src_points.len()));
                    }
                }

                let mut dst_h8_vec: Vec<CellIndex> = dest_h8_set.into_iter().collect();
                dst_h8_vec.sort();
                let mut dst_points: Vec<RoadPoint> = Vec::new();
                let mut dst_ranges: HashMap<CellIndex, std::ops::Range<usize>> =
                    HashMap::with_capacity(dst_h8_vec.len());
                for &dh8 in &dst_h8_vec {
                    if let Some(pts) = rep_by_h8.get(&dh8) {
                        let start = dst_points.len();
                        dst_points.extend_from_slice(pts);
                        dst_ranges.insert(dh8, start..dst_points.len());
                    }
                }

                if let Ok(matrix) = query_osrm_table(client, &osrm_url, &src_points, &dst_points) {
                    for (orig_h8, s_range) in &src_ranges {
                        for dest_h8 in orig_h8.grid_disk::<Vec<_>>(TIER1_RES8_MAX_RINGS as u32) {
                            if dest_h8 == *orig_h8 {
                                continue;
                            }
                            if let Some(d_range) = dst_ranges.get(&dest_h8) {
                                let mut samples = Vec::with_capacity(12);
                                for si in s_range.clone() {
                                    if let Some(row) = matrix.get(si) {
                                        for di in d_range.clone() {
                                            if let Some(Some(dur)) = row.get(di) {
                                                if *dur > 1.0 && dur.is_finite() {
                                                    samples.push(*dur);
                                                }
                                            }
                                        }
                                    }
                                }
                                if !samples.is_empty() {
                                    local_pairs.push((canonical_pair(*orig_h8, dest_h8), samples));
                                }
                            }
                        }
                    }
                }
                let done =
                    completed_batches.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if done % 100 == 0 || done == total_batches {
                    println!(
                        "[build]   Tier 0/1 progress: {}/{} Res-6 batches ({:.1}s)",
                        done,
                        total_batches,
                        t_t1.elapsed().as_secs_f64()
                    );
                }
                local_pairs
            },
        )
        .flatten()
        .collect();

    // Pool bidirectional samples (A -> B and B -> A) into each canonical undirected pair (min(A,B), max(A,B))
    let mut tier01_pooled: HashMap<(CellIndex, CellIndex), Vec<f32>> =
        HashMap::with_capacity(tier01_raw_samples.len() / 2 + 1024);
    for (pair, samples) in tier01_raw_samples {
        tier01_pooled.entry(pair).or_default().extend(samples);
    }
    let tier01_records: Vec<((CellIndex, CellIndex), TransitSummary)> = tier01_pooled
        .into_par_iter()
        .filter_map(|(pair, mut samples)| {
            TransitSummary::from_samples(&mut samples).map(|s| (pair, s))
        })
        .collect();

    println!(
        "[build] Tier 0 & Tier 1 complete in {:.1}s: {} canonical undirected Res-8 pair distributions.",
        t_t1.elapsed().as_secs_f64(),
        tier01_records.len()
    );

    // Step 2: Compute Tier 2 (Res 7 <-> Res 7, d7 <= 10)
    let t_t2 = Instant::now();
    let mut core_h7_set: HashSet<CellIndex> = HashSet::new();
    for &h8 in &core_h8_cells {
        if let Some(h7) = h8.parent(Resolution::Seven) {
            core_h7_set.insert(h7);
        }
    }
    let mut core_by_h5: BTreeMap<CellIndex, Vec<CellIndex>> = BTreeMap::new();
    for &h7 in &core_h7_set {
        if let Some(h5) = h7.parent(Resolution::Five) {
            core_by_h5.entry(h5).or_default().push(h7);
        }
    }
    let h5_groups_for_t2: Vec<(CellIndex, Vec<CellIndex>)> = core_by_h5.into_iter().collect();

    let tier2_raw: Vec<((CellIndex, CellIndex), Vec<f32>)> = h5_groups_for_t2
        .par_iter()
        .map_init(
            || {
                Client::builder()
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                    .unwrap()
            },
            |client, &(orig_h5, ref orig_h7_list)| {
                let mut dest_h7_set: HashSet<CellIndex> = HashSet::new();
                for &orig_h7 in orig_h7_list {
                    for ring_cell in orig_h7.grid_disk::<Vec<_>>(TIER2_RES7_MAX_RINGS as u32) {
                        if ring_cell != orig_h7 && rep_by_h7.contains_key(&ring_cell) {
                            let include = if core_h7_set.contains(&ring_cell) {
                                ring_cell
                                    .parent(Resolution::Five)
                                    .map(|dh5| u64::from(dh5) >= u64::from(orig_h5))
                                    .unwrap_or(true)
                            } else {
                                true
                            };
                            if include {
                                dest_h7_set.insert(ring_cell);
                            }
                        }
                    }
                }

                let mut src_pts: Vec<(CellIndex, RoadPoint)> = Vec::new();
                for &oh7 in orig_h7_list {
                    if let Some(pts) = rep_by_h7.get(&oh7) {
                        for &p in pts.iter().take(4) {
                            src_pts.push((oh7, p));
                        }
                    }
                }

                let mut dst_pts: Vec<(CellIndex, RoadPoint)> = Vec::new();
                for &dh7 in &dest_h7_set {
                    if let Some(pts) = rep_by_h7.get(&dh7) {
                        for &p in pts.iter().take(4) {
                            dst_pts.push((dh7, p));
                        }
                    }
                }

                let raw_src: Vec<RoadPoint> = src_pts.iter().map(|x| x.1).collect();
                let raw_dst: Vec<RoadPoint> = dst_pts.iter().map(|x| x.1).collect();

                let mut local_pairs: HashMap<(CellIndex, CellIndex), Vec<f32>> = HashMap::new();
                if let Ok(matrix) = query_osrm_table(client, &osrm_url, &raw_src, &raw_dst) {
                    for (si, (oh7, _)) in src_pts.iter().enumerate() {
                        if let Some(row) = matrix.get(si) {
                            for (di, (dh7, _)) in dst_pts.iter().enumerate() {
                                if oh7 != dh7 {
                                    if let Ok(d7) = oh7.grid_distance(*dh7) {
                                        if d7 <= TIER2_RES7_MAX_RINGS {
                                            if let Some(Some(dur)) = row.get(di) {
                                                if *dur > 1.0 && dur.is_finite() {
                                                    local_pairs
                                                        .entry(canonical_pair(*oh7, *dh7))
                                                        .or_default()
                                                        .push(*dur);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                local_pairs.into_iter().collect::<Vec<_>>()
            },
        )
        .flatten()
        .collect();

    let mut tier2_pooled: HashMap<(CellIndex, CellIndex), Vec<f32>> = HashMap::new();
    for (pair, samples) in tier2_raw {
        tier2_pooled.entry(pair).or_default().extend(samples);
    }
    let tier2_records: Vec<((CellIndex, CellIndex), TransitSummary)> = tier2_pooled
        .into_par_iter()
        .filter_map(|(pair, mut samples)| {
            TransitSummary::from_samples(&mut samples).map(|s| (pair, s))
        })
        .collect();

    println!(
        "[build] Tier 2 complete in {:.1}s: {} canonical undirected Res-7 pair distributions.",
        t_t2.elapsed().as_secs_f64(),
        tier2_records.len()
    );

    // Step 3: Compute Tier 3 (Res 6 <-> Res 6, d6 <= 12) & Tier 4 (Res 5 <-> Res 5, d5 <= 14)
    let t_t34 = Instant::now();
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let mut core_h6_list: Vec<CellIndex> = core_h7_set
        .iter()
        .filter_map(|h7| h7.parent(Resolution::Six))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    core_h6_list.sort();

    let core_h6_set: HashSet<CellIndex> = core_h6_list.iter().copied().collect();
    let tier3_raw: Vec<((CellIndex, CellIndex), Vec<f32>)> = core_h6_list
        .par_chunks(20)
        .map_init(
            || {
                Client::builder()
                    .timeout(std::time::Duration::from_secs(120))
                    .build()
                    .unwrap()
            },
            |client, chunk| {
                let min_chunk_h6 = chunk.first().copied().unwrap();
                let mut dst_h6_set: HashSet<CellIndex> = HashSet::new();
                for &oh6 in chunk {
                    for ring_cell in oh6.grid_disk::<Vec<_>>(TIER3_RES6_MAX_RINGS as u32) {
                        if ring_cell != oh6 && rep_by_h6.contains_key(&ring_cell) {
                            if !core_h6_set.contains(&ring_cell)
                                || u64::from(ring_cell) >= u64::from(min_chunk_h6)
                            {
                                dst_h6_set.insert(ring_cell);
                            }
                        }
                    }
                }
                let mut src_pts: Vec<(CellIndex, RoadPoint)> = Vec::new();
                for &oh6 in chunk {
                    if let Some(pts) = rep_by_h6.get(&oh6) {
                        for &p in pts.iter().take(5) {
                            src_pts.push((oh6, p));
                        }
                    }
                }
                let mut dst_pts: Vec<(CellIndex, RoadPoint)> = Vec::new();
                for &dh6 in &dst_h6_set {
                    if let Some(pts) = rep_by_h6.get(&dh6) {
                        for &p in pts.iter().take(5) {
                            dst_pts.push((dh6, p));
                        }
                    }
                }
                let raw_src: Vec<RoadPoint> = src_pts.iter().map(|x| x.1).collect();
                let raw_dst: Vec<RoadPoint> = dst_pts.iter().map(|x| x.1).collect();
                let mut local_pairs: HashMap<(CellIndex, CellIndex), Vec<f32>> = HashMap::new();
                if let Ok(matrix) = query_osrm_table(client, &osrm_url, &raw_src, &raw_dst) {
                    for (si, (oh6, _)) in src_pts.iter().enumerate() {
                        if let Some(row) = matrix.get(si) {
                            for (di, (dh6, _)) in dst_pts.iter().enumerate() {
                                if oh6 != dh6 {
                                    if let Ok(d6) = oh6.grid_distance(*dh6) {
                                        if d6 <= TIER3_RES6_MAX_RINGS {
                                            if let Some(Some(dur)) = row.get(di) {
                                                if *dur > 1.0 && dur.is_finite() {
                                                    local_pairs
                                                        .entry(canonical_pair(*oh6, *dh6))
                                                        .or_default()
                                                        .push(*dur);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                local_pairs.into_iter().collect::<Vec<_>>()
            },
        )
        .flatten()
        .collect();

    let mut tier3_pooled: HashMap<(CellIndex, CellIndex), Vec<f32>> = HashMap::new();
    for (pair, samples) in tier3_raw {
        tier3_pooled.entry(pair).or_default().extend(samples);
    }

    // Tier 4: Res 5 <-> Res 5 (d5 <= 14)
    let mut all_h5_cells: Vec<CellIndex> = rep_by_h5.keys().copied().collect();
    all_h5_cells.sort();
    let mut h5_pts: Vec<(CellIndex, RoadPoint)> = Vec::new();
    for &h5 in &all_h5_cells {
        if let Some(pts) = rep_by_h5.get(&h5) {
            for &p in pts {
                h5_pts.push((h5, p));
            }
        }
    }
    let raw_h5: Vec<RoadPoint> = h5_pts.iter().map(|x| x.1).collect();
    let mut tier4_pooled: HashMap<(CellIndex, CellIndex), Vec<f32>> = HashMap::new();
    match query_osrm_table(&client, &osrm_url, &raw_h5, &raw_h5) {
        Ok(matrix) => {
            for (si, (oh5, _)) in h5_pts.iter().enumerate() {
                if let Some(row) = matrix.get(si) {
                    for (di, (dh5, _)) in h5_pts.iter().enumerate() {
                        if oh5 != dh5 {
                            if let Ok(d5) = oh5.grid_distance(*dh5) {
                                if d5 <= TIER4_RES5_MAX_RINGS {
                                    if let Some(Some(dur)) = row.get(di) {
                                        if *dur > 1.0 && dur.is_finite() {
                                            tier4_pooled
                                                .entry(canonical_pair(*oh5, *dh5))
                                                .or_default()
                                                .push(*dur);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => eprintln!("[build] Tier 4 batch error: {}", e),
    }

    let mut tier34_records: Vec<((CellIndex, CellIndex), TransitSummary)> = Vec::new();
    for (pair, mut samples) in tier3_pooled {
        if let Some(s) = TransitSummary::from_samples(&mut samples) {
            tier34_records.push((pair, s));
        }
    }
    let tier3_count = tier34_records.len();
    for (pair, mut samples) in tier4_pooled {
        if let Some(s) = TransitSummary::from_samples(&mut samples) {
            tier34_records.push((pair, s));
        }
    }
    let tier4_count = tier34_records.len() - tier3_count;

    println!(
        "[build] Tier 3 ({} Res-6 pairs) & Tier 4 ({} Res-5 pairs) complete in {:.1}s.",
        tier3_count,
        tier4_count,
        t_t34.elapsed().as_secs_f64()
    );

    // Step 4: Write Packed CSR Binary Index (.h3t)
    let t_db = Instant::now();
    let mut all_records = Vec::with_capacity(
        tier01_records.len() + tier2_records.len() + tier34_records.len(),
    );
    all_records.extend(tier01_records);
    all_records.extend(tier2_records);
    all_records.extend(tier34_records);

    let (num_origins, num_edges) = H3CsrIndex::write_to_path(&db_path, &all_records)?;
    let file_size_mb = std::fs::metadata(&db_path)?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "[build] Wrote Packed CSR Index ({} unique origins, {} canonical edges) into {:?} ({:.2} MB) in {:.2}s!",
        num_origins,
        num_edges,
        db_path,
        file_size_mb,
        t_db.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_backtest(
    pbf: PathBuf,
    db_path: PathBuf,
    osrm_url: String,
    core_bbox_str: String,
    pairs_per_tier: usize,
) -> Result<()> {
    let bbox = BBox::parse(&core_bbox_str)?;
    let by_h8 = extract_road_points_by_res8(&pbf)?;
    let csr_index = H3CsrIndex::open_mmap(&db_path)?;

    let mut core_h8_cells: Vec<CellIndex> = by_h8
        .keys()
        .copied()
        .filter(|&h8| {
            let ll = LatLng::from(h8);
            bbox.contains(ll.lat(), ll.lng())
        })
        .collect();
    core_h8_cells.sort();

    let mut rng = rand::rngs::StdRng::seed_from_u64(20261106);

    println!(
        "\n=== H3-Transit Packed CSR Index ({} origins, {} edges) PIT Holdout Backtest ===",
        csr_index.num_origins(),
        csr_index.num_edges()
    );

    let model_names = [
        "Shifted LogNormal (MoM)",
        "Shifted LogNormal (Quantile)",
        "Shifted Gamma (MoM)",
        "Shifted Weibull (Quantile)",
        "Gaussian (Normal) Baseline",
    ];

    let mut all_pits_by_model: Vec<Vec<f64>> = vec![Vec::new(); model_names.len()];

    for tier in [
        DistanceTier::Tier0IntraRes8,
        DistanceTier::Tier1LocalRes8,
        DistanceTier::Tier2MidRes7,
        DistanceTier::Tier3RegionalRes6,
        DistanceTier::Tier4LongRes5,
    ] {
        let mut sampled_pairs: Vec<(RoadPoint, RoadPoint, TransitSummary)> =
            Vec::with_capacity(pairs_per_tier);
        let mut attempts = 0usize;

        while sampled_pairs.len() < pairs_per_tier && attempts < pairs_per_tier * 60 {
            attempts += 1;
            let orig_h8 = *core_h8_cells.choose(&mut rng).unwrap();
            let dest_h8 = match tier {
                DistanceTier::Tier0IntraRes8 => orig_h8,
                DistanceTier::Tier1LocalRes8 => {
                    let disk: Vec<CellIndex> = orig_h8
                        .grid_disk::<Vec<_>>(TIER1_RES8_MAX_RINGS as u32)
                        .into_iter()
                        .filter(|&c| c != orig_h8 && by_h8.contains_key(&c))
                        .collect();
                    if disk.is_empty() {
                        continue;
                    }
                    *disk.choose(&mut rng).unwrap()
                }
                _ => {
                    let cand = *core_h8_cells.choose(&mut rng).unwrap();
                    if let Some((resolved_tier, _, _)) = resolve_lookup_pair(orig_h8, cand) {
                        if resolved_tier == tier {
                            cand
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    }
                }
            };

            let Some(orig_pts) = by_h8.get(&orig_h8) else {
                continue;
            };
            let Some(dest_pts) = by_h8.get(&dest_h8) else {
                continue;
            };
            if tier == DistanceTier::Tier0IntraRes8 && orig_pts.len() < 4 {
                continue;
            }
            let p_orig = *orig_pts.choose(&mut rng).unwrap();
            let p_dest = *dest_pts.choose(&mut rng).unwrap();
            if (p_orig.lat - p_dest.lat).abs() < 1e-5 && (p_orig.lon - p_dest.lon).abs() < 1e-5 {
                continue;
            }

            if let Some((resolved_tier, summary)) = csr_index.get_pair_res8(orig_h8, dest_h8) {
                if resolved_tier == tier {
                    sampled_pairs.push((p_orig, p_dest, summary));
                }
            }
        }

        let eval_results: Vec<(f64, [f64; 5])> = sampled_pairs
            .par_iter()
            .map_init(
                || {
                    Client::builder()
                        .timeout(std::time::Duration::from_secs(30))
                        .build()
                        .unwrap()
                },
                |client, (p_orig, p_dest, summary)| {
                    if let Some(true_sec) =
                        query_osrm_single_route(client, &osrm_url, *p_orig, *p_dest)
                    {
                        let t = true_sec as f64;
                        if t > 1.0 && t.is_finite() {
                            let rel_err = ((summary.mean_sec as f64) - t).abs() / t;
                            let models = [
                                ParametricModel::shifted_lognormal_mom(summary),
                                ParametricModel::shifted_lognormal_quantile(summary),
                                ParametricModel::shifted_gamma(summary),
                                ParametricModel::shifted_weibull(summary),
                                ParametricModel::gaussian_baseline(summary),
                            ];
                            return Some((
                                rel_err,
                                [
                                    models[0].cdf(t),
                                    models[1].cdf(t),
                                    models[2].cdf(t),
                                    models[3].cdf(t),
                                    models[4].cdf(t),
                                ],
                            ));
                        }
                    }
                    None
                },
            )
            .flatten()
            .collect();

        let mut tier_pits: Vec<Vec<f64>> = vec![Vec::with_capacity(eval_results.len()); 5];
        let mut rel_errors: Vec<f64> = Vec::with_capacity(eval_results.len());
        for (re, pits) in eval_results {
            rel_errors.push(re);
            for m_idx in 0..5 {
                tier_pits[m_idx].push(pits[m_idx]);
                all_pits_by_model[m_idx].push(pits[m_idx]);
            }
        }

        let n_eval = tier_pits[0].len().max(1);
        let mean_ape =
            100.0 * (rel_errors.iter().sum::<f64>() / (rel_errors.len().max(1) as f64));
        println!(
            "\n--- {} (N = {}, Mean Mean-vs-Point Spread = {:.1}%) ---",
            tier.label(),
            n_eval,
            mean_ape
        );
        println!(
            "{:<30} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7}",
            "Distribution Family", "P<=10%", "P<=25%", "P<=50%", "P<=75%", "P<=90%", "KS Stat"
        );
        println!("{}", "-".repeat(88));
        for (m_idx, name) in model_names.iter().enumerate() {
            let stats = compute_pit_metrics(&mut tier_pits[m_idx]);
            println!(
                "{:<30} | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% | {:>7.4}",
                name,
                stats.p10 * 100.0,
                stats.p25 * 100.0,
                stats.p50 * 100.0,
                stats.p75 * 100.0,
                stats.p90 * 100.0,
                stats.ks
            );
        }
    }

    println!(
        "\n=== OVERALL CALIBRATION ACROSS ALL TIERS (N = {}) ===",
        all_pits_by_model[0].len()
    );
    println!(
        "{:<30} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7}",
        "Distribution Family (Ideal ->)", "10.0%", "25.0%", "50.0%", "75.0%", "90.0%", "KS Stat"
    );
    println!("{}", "=".repeat(88));
    for (m_idx, name) in model_names.iter().enumerate() {
        let stats = compute_pit_metrics(&mut all_pits_by_model[m_idx]);
        println!(
            "{:<30} | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% | {:>7.4}",
            name,
            stats.p10 * 100.0,
            stats.p25 * 100.0,
            stats.p50 * 100.0,
            stats.p75 * 100.0,
            stats.p90 * 100.0,
            stats.ks
        );
    }

    Ok(())
}

struct PitStats {
    p10: f64,
    p25: f64,
    p50: f64,
    p75: f64,
    p90: f64,
    ks: f64,
}

fn compute_pit_metrics(u: &mut [f64]) -> PitStats {
    if u.is_empty() {
        return PitStats {
            p10: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            p90: 0.0,
            ks: 1.0,
        };
    }
    u.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = u.len() as f64;
    let p10 = u.iter().filter(|&&x| x <= 0.10).count() as f64 / n;
    let p25 = u.iter().filter(|&&x| x <= 0.25).count() as f64 / n;
    let p50 = u.iter().filter(|&&x| x <= 0.50).count() as f64 / n;
    let p75 = u.iter().filter(|&&x| x <= 0.75).count() as f64 / n;
    let p90 = u.iter().filter(|&&x| x <= 0.90).count() as f64 / n;

    let mut ks = 0.0f64;
    for (i, &val) in u.iter().enumerate() {
        let emp_hi = ((i + 1) as f64) / n;
        let emp_lo = (i as f64) / n;
        ks = ks.max((emp_hi - val).abs()).max((val - emp_lo).abs());
    }
    PitStats {
        p10,
        p25,
        p50,
        p75,
        p90,
        ks,
    }
}

fn run_query(db_path: PathBuf, orig_str: String, dest_str: String) -> Result<()> {
    let orig: CellIndex = orig_str.parse().context("Invalid origin H3 index")?;
    let dest: CellIndex = dest_str.parse().context("Invalid destination H3 index")?;
    let csr_index = H3CsrIndex::open_mmap(&db_path)?;

    // Warm + 1000-iteration average latency benchmark
    let _ = csr_index.get_pair_res8(orig, dest);
    let t0 = Instant::now();
    let mut res = None;
    for _ in 0..1000 {
        res = std::hint::black_box(csr_index.get_pair_res8(
            std::hint::black_box(orig),
            std::hint::black_box(dest),
        ));
    }
    let avg_ns = t0.elapsed().as_nanos() / 1000;

    if let Some((tier, summary)) = res {
        let (_, k_orig, k_dest) = resolve_lookup_pair(orig, dest).unwrap();
        println!("Lookup latency : {} ns/query (1000-iter avg)", avg_ns);
        println!("Resolved Tier  : {}", tier.label());
        println!("Canonical Pair : {} <-> {}", k_orig, k_dest);
        println!(
            "Summary (10B)  : min={}s, p50={}s, mean={}s, p90={}s, stdev={}s",
            summary.min_sec, summary.p50_sec, summary.mean_sec, summary.p90_sec, summary.stdev_sec
        );
    } else {
        println!("No route record found for {} -> {}", orig, dest);
    }
    Ok(())
}

fn run_merge(output: PathBuf, inputs: Vec<PathBuf>) -> Result<()> {
    let valid_inputs: Vec<PathBuf> = inputs.into_iter().filter(|p| p.exists()).collect();
    anyhow::ensure!(!valid_inputs.is_empty(), "No existing input .h3t files found");
    println!(
        "[merge] Merging {} regional .h3t shards into {:?}...",
        valid_inputs.len(),
        output
    );
    let t0 = Instant::now();
    let (num_origins, num_edges) = H3CsrIndex::merge_files(&valid_inputs, &output)?;
    let size_mb = std::fs::metadata(&output)?.len() as f64 / (1024.0 * 1024.0);
    println!(
        "[merge] Complete in {:.2}s: {} unique origins, {} canonical edges ({:.2} MB).",
        t0.elapsed().as_secs_f64(),
        num_origins,
        num_edges,
        size_mb
    );
    Ok(())
}

/// Runs `osmium extract -c` to extract a batch of `(id, core_bbox, out_pbf_path)` items
/// from `source_pbf` in a single streaming pass, padding each `core_bbox` by `buffer_deg`.
fn osmium_extract_batch(
    source_pbf: &PathBuf,
    work_dir: &PathBuf,
    items: &[(String, BBox, PathBuf)],
    buffer_deg: f64,
) -> Result<()> {
    let to_extract: Vec<&(String, BBox, PathBuf)> =
        items.iter().filter(|(_, _, p)| !p.exists()).collect();
    if to_extract.is_empty() {
        return Ok(());
    }

    let mut extracts_json = Vec::new();
    for (_, core_bbox, out_pbf) in &to_extract {
        let padded = core_bbox.padded(buffer_deg);
        let file_name = out_pbf
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        extracts_json.push(serde_json::json!({
            "output": file_name,
            "output_format": "pbf",
            "bbox": [padded.min_lon, padded.min_lat, padded.max_lon, padded.max_lat]
        }));
    }

    let cfg_dir = to_extract[0].2.parent().unwrap_or(work_dir.as_path());
    let cfg_path = cfg_dir.join("osmium_batch_extract.json");
    let cfg_doc = serde_json::json!({
        "directory": cfg_dir.to_string_lossy(),
        "extracts": extracts_json
    });
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg_doc)?)?;

    println!(
        "[osmium] Extracting {} tile(s) from {:?}...",
        to_extract.len(),
        source_pbf
    );
    let status = std::process::Command::new("osmium")
        .args([
            "extract",
            "-c",
            &cfg_path.to_string_lossy(),
            "-s",
            "simple",
            "--overwrite",
            &source_pbf.to_string_lossy(),
        ])
        .status()
        .context("Failed to execute osmium extract")?;
    let _ = std::fs::remove_file(&cfg_path);
    anyhow::ensure!(status.success(), "osmium extract exited with {}", status);
    Ok(())
}

fn find_car_lua_profile() -> PathBuf {
    let candidates = [
        "/opt/homebrew/opt/osrm-backend/share/osrm-backend/profiles/car.lua",
        "/usr/local/opt/osrm-backend/share/osrm-backend/profiles/car.lua",
        "/usr/share/osrm/profiles/car.lua",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return p;
        }
    }
    PathBuf::from("car.lua")
}

/// Recursively processes `(tile_id, core_bbox, pbf_path)`:
/// - If `tiles_dir/{tile_id}.h3t` (or any split descendants) already exist, skips.
/// - If `pbf_path` > `max_tile_pbf_mb`, splits `core_bbox` in half, extracts both halves from `pbf_path`
///   in a single `osmium extract -c` call, deletes `pbf_path`, and recurses.
/// - Otherwise builds OSRM graph, runs `run_build`, writes `tiles_dir/{tile_id}.h3t`, and deletes `.osrm*` and `.osm.pbf`.
fn process_tile_recursive(
    tile_id: &str,
    core_bbox: BBox,
    pbf_path: PathBuf,
    pbf_dir: &PathBuf,
    tiles_dir: &PathBuf,
    max_tile_pbf_mb: u64,
    buffer_deg: f64,
    osrm_port: u16,
) -> Result<Vec<PathBuf>> {
    let leaf_h3t = tiles_dir.join(format!("{}.h3t", tile_id));
    let done_marker = tiles_dir.join(format!("{}.done", tile_id));
    if leaf_h3t.exists() {
        println!(
            "[build-us] Leaf tile '{}' already complete ({:?}), skipping.",
            tile_id, leaf_h3t
        );
        let _ = std::fs::remove_file(&pbf_path);
        return Ok(vec![leaf_h3t]);
    }
    if done_marker.exists() {
        println!(
            "[build-us] Split subtree '{}' already complete, skipping.",
            tile_id
        );
        let _ = std::fs::remove_file(&pbf_path);
        return Ok(Vec::new());
    }

    let (bbox_a, bbox_b) = core_bbox.split_half();
    let id_a = format!("{}_a", tile_id);
    let id_b = format!("{}_b", tile_id);
    let pbf_a = pbf_dir.join(format!("{}.osm.pbf", id_a));
    let pbf_b = pbf_dir.join(format!("{}.osm.pbf", id_b));

    // If this parent tile was already bisected in a prior run and its parent PBF was deleted,
    // resume directly on its children!
    if !pbf_path.exists()
        && (has_any_progress(pbf_dir, tiles_dir, &id_a)
            || has_any_progress(pbf_dir, tiles_dir, &id_b))
    {
        println!(
            "[build-us] Resuming already-bisected subtree '{}'...",
            tile_id
        );
        let mut out = Vec::new();
        out.extend(process_tile_recursive(
            &id_a,
            bbox_a,
            pbf_a,
            pbf_dir,
            tiles_dir,
            max_tile_pbf_mb,
            buffer_deg,
            osrm_port,
        )?);
        out.extend(process_tile_recursive(
            &id_b,
            bbox_b,
            pbf_b,
            pbf_dir,
            tiles_dir,
            max_tile_pbf_mb,
            buffer_deg,
            osrm_port,
        )?);
        std::fs::write(&done_marker, b"ok")?;
        return Ok(out);
    }

    let pbf_bytes = std::fs::metadata(&pbf_path)
        .with_context(|| format!("Missing tile PBF {:?}", pbf_path))?
        .len();
    let pbf_mb = pbf_bytes / (1024 * 1024);

    if pbf_bytes < 50_000 {
        println!("[build-us] Tile '{}' PBF is empty ({} bytes), skipping.", tile_id, pbf_bytes);
        let _ = std::fs::remove_file(&pbf_path);
        return Ok(Vec::new());
    }

    let min_span = (core_bbox.max_lon - core_bbox.min_lon)
        .max(core_bbox.max_lat - core_bbox.min_lat);

    if pbf_mb > max_tile_pbf_mb && min_span > 2.2 {
        println!(
            "[build-us] Tile '{}' PBF is {} MB (> {} MB threshold, span={:.2} deg) -> bisecting into 2 sub-tiles...",
            tile_id, pbf_mb, max_tile_pbf_mb, min_span
        );
        let (bbox_a, bbox_b) = core_bbox.split_half();
        let id_a = format!("{}_a", tile_id);
        let id_b = format!("{}_b", tile_id);
        let pbf_a = pbf_dir.join(format!("{}.osm.pbf", id_a));
        let pbf_b = pbf_dir.join(format!("{}.osm.pbf", id_b));

        let mut needed = Vec::new();
        if !is_tile_complete(tiles_dir, &id_a) && !pbf_a.exists() {
            needed.push((id_a.clone(), bbox_a, pbf_a.clone()));
        }
        if !is_tile_complete(tiles_dir, &id_b) && !pbf_b.exists() {
            needed.push((id_b.clone(), bbox_b, pbf_b.clone()));
        }

        if !needed.is_empty() {
            osmium_extract_batch(&pbf_path, pbf_dir, &needed, buffer_deg)?;
        }
        // Remove parent PBF immediately so disk usage stays bounded
        let _ = std::fs::remove_file(&pbf_path);

        let mut out = Vec::new();
        out.extend(process_tile_recursive(
            &id_a,
            bbox_a,
            pbf_a,
            pbf_dir,
            tiles_dir,
            max_tile_pbf_mb,
            buffer_deg,
            osrm_port,
        )?);
        out.extend(process_tile_recursive(
            &id_b,
            bbox_b,
            pbf_b,
            pbf_dir,
            tiles_dir,
            max_tile_pbf_mb,
            buffer_deg,
            osrm_port,
        )?);
        std::fs::write(&done_marker, b"ok")?;
        return Ok(out);
    }

    println!(
        "\n================================================================================\n\
         [build-us] Building leaf tile '{}' ({} MB PBF, core bbox: {})\n\
         ================================================================================",
        tile_id,
        pbf_mb,
        core_bbox.to_arg_string()
    );

    let car_lua = find_car_lua_profile();
    let osrm_base = pbf_dir.join(format!("{}.osrm", tile_id));

    // 1. osrm-extract -> osrm-partition -> osrm-customize
    let t_osrm = Instant::now();
    let st1 = std::process::Command::new("osrm-extract")
        .args([
            "-p",
            &car_lua.to_string_lossy(),
            &pbf_path.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .status()
        .context("Failed to run osrm-extract")?;
    anyhow::ensure!(st1.success(), "osrm-extract failed on {}", tile_id);

    let st2 = std::process::Command::new("osrm-partition")
        .arg(&osrm_base.to_string_lossy().as_ref())
        .stdout(std::process::Stdio::null())
        .status()
        .context("Failed to run osrm-partition")?;
    anyhow::ensure!(st2.success(), "osrm-partition failed on {}", tile_id);

    let st3 = std::process::Command::new("osrm-customize")
        .arg(&osrm_base.to_string_lossy().as_ref())
        .stdout(std::process::Stdio::null())
        .status()
        .context("Failed to run osrm-customize")?;
    anyhow::ensure!(st3.success(), "osrm-customize failed on {}", tile_id);

    println!(
        "[build-us] OSRM MLD graph prepared for '{}' in {:.1}s.",
        tile_id,
        t_osrm.elapsed().as_secs_f64()
    );

    // 2. Start osrm-routed on osrm_port
    let mut osrm_child = std::process::Command::new("osrm-routed")
        .args([
            "--algorithm",
            "mld",
            "-p",
            &osrm_port.to_string(),
            "--max-table-size",
            "10000",
            &osrm_base.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("Failed to spawn osrm-routed")?;

    let osrm_url = format!("http://127.0.0.1:{}", osrm_port);
    std::thread::sleep(std::time::Duration::from_millis(1500));

    // 3. Run H3 distillation for this tile
    let tmp_h3t = tiles_dir.join(format!("{}.h3t.tmp", tile_id));
    let build_res = run_build(
        pbf_path.clone(),
        tmp_h3t.clone(),
        osrm_url,
        core_bbox.to_arg_string(),
        3,
    );

    // 4. Stop osrm-routed and clean up all temporary .osrm* and .osm.pbf files
    let _ = osrm_child.kill();
    let _ = osrm_child.wait();
    if let Ok(entries) = std::fs::read_dir(pbf_dir) {
        let prefix = format!("{}.osrm", tile_id);
        for entry in entries.flatten() {
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(&prefix)
            {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let _ = std::fs::remove_file(&pbf_path);

    build_res?;
    std::fs::rename(&tmp_h3t, &leaf_h3t)?;
    Ok(vec![leaf_h3t])
}

fn is_tile_complete(tiles_dir: &PathBuf, tile_id: &str) -> bool {
    tiles_dir.join(format!("{}.h3t", tile_id)).exists()
        || tiles_dir.join(format!("{}.done", tile_id)).exists()
}

fn has_any_progress(pbf_dir: &PathBuf, tiles_dir: &PathBuf, prefix: &str) -> bool {
    if is_tile_complete(tiles_dir, prefix) || pbf_dir.join(format!("{}.osm.pbf", prefix)).exists() {
        return true;
    }
    let pat = format!("{}_", prefix);
    for dir in [pbf_dir, tiles_dir] {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                if e.file_name().to_string_lossy().starts_with(&pat) {
                    return true;
                }
            }
        }
    }
    false
}

fn run_build_us(
    us_pbf: PathBuf,
    work_dir: PathBuf,
    output: PathBuf,
    max_tile_pbf_mb: u64,
    osrm_port: u16,
) -> Result<()> {
    let pbf_dir = work_dir.join("pbf");
    let tiles_dir = work_dir.join("tiles");
    std::fs::create_dir_all(&pbf_dir)?;
    std::fs::create_dir_all(&tiles_dir)?;

    let buffer_deg = 0.65; // ~70 km overlap buffer around every tile

    // 10 non-overlapping core regions partitioning all 50 US states + DC + PR
    let macro_regions: Vec<(&str, BBox)> = vec![
        (
            "r01_northeast",
            BBox {
                min_lon: -80.6,
                min_lat: 38.8,
                max_lon: -66.8,
                max_lat: 47.6,
            },
        ),
        (
            "r02_southeast",
            BBox {
                min_lon: -88.5,
                min_lat: 24.4,
                max_lon: -75.0,
                max_lat: 38.8,
            },
        ),
        (
            "r03_great_lakes",
            BBox {
                min_lon: -90.5,
                min_lat: 36.5,
                max_lon: -80.6,
                max_lat: 49.4,
            },
        ),
        (
            "r04_plains_midwest",
            BBox {
                min_lon: -104.0,
                min_lat: 36.5,
                max_lon: -90.5,
                max_lat: 49.4,
            },
        ),
        (
            "r05_south_central",
            BBox {
                min_lon: -106.7,
                min_lat: 25.8,
                max_lon: -88.5,
                max_lat: 36.5,
            },
        ),
        (
            "r06_mountain_west",
            BBox {
                min_lon: -117.3,
                min_lat: 31.3,
                max_lon: -104.0,
                max_lat: 49.0,
            },
        ),
        (
            "r07_pacific_coast",
            BBox {
                min_lon: -124.8,
                min_lat: 32.5,
                max_lon: -117.3,
                max_lat: 49.0,
            },
        ),
        (
            "r08_alaska",
            BBox {
                min_lon: -170.0,
                min_lat: 51.0,
                max_lon: -129.0,
                max_lat: 71.5,
            },
        ),
        (
            "r09_hawaii",
            BBox {
                min_lon: -160.6,
                min_lat: 18.8,
                max_lon: -154.5,
                max_lat: 22.4,
            },
        ),
        (
            "r10_puerto_rico",
            BBox {
                min_lon: -67.5,
                min_lat: 17.8,
                max_lon: -64.5,
                max_lat: 18.6,
            },
        ),
    ];

    // Process macro-regions in batches of 3 so we only stream `us-261103.osm.pbf` at most 4 times
    // and never hold more than 3 macro PBFs on disk at once!
    for group in macro_regions.chunks(3) {
        let mut batch_to_extract = Vec::new();
        for &(id, bbox) in group {
            let pbf_path = pbf_dir.join(format!("{}.osm.pbf", id));
            if !has_any_progress(&pbf_dir, &tiles_dir, id) {
                batch_to_extract.push((id.to_string(), bbox, pbf_path));
            }
        }
        if !batch_to_extract.is_empty() {
            osmium_extract_batch(&us_pbf, &pbf_dir, &batch_to_extract, buffer_deg)?;
        }

        for &(id, bbox) in group {
            if is_tile_complete(&tiles_dir, id) {
                println!(
                    "[build-us] Macro-region '{}' already complete, skipping.",
                    id
                );
                continue;
            }
            let pbf_path = pbf_dir.join(format!("{}.osm.pbf", id));
            process_tile_recursive(
                id,
                bbox,
                pbf_path,
                &pbf_dir,
                &tiles_dir,
                max_tile_pbf_mb,
                buffer_deg,
                osrm_port,
            )?;
        }
    }

    // Collect all completed .h3t shards in tiles_dir and merge into `output`
    let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(&tiles_dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("h3t"))
        .collect();
    shard_paths.sort();

    run_merge(output, shard_paths)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Build {
            pbf,
            db,
            osrm_url,
            core_bbox,
            samples_per_res8,
        } => run_build(pbf, db, osrm_url, core_bbox, samples_per_res8),
        Commands::Backtest {
            pbf,
            db,
            osrm_url,
            core_bbox,
            pairs_per_tier,
        } => run_backtest(pbf, db, osrm_url, core_bbox, pairs_per_tier),
        Commands::Query { db, orig, dest } => run_query(db, orig, dest),
        Commands::Merge { output, inputs } => run_merge(output, inputs),
        Commands::BuildUs {
            us_pbf,
            work_dir,
            output,
            max_tile_pbf_mb,
            osrm_port,
        } => run_build_us(us_pbf, work_dir, output, max_tile_pbf_mb, osrm_port),
    }
}
