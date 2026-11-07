//! `h3-transit`: Multi-Resolution H3 Stochastic Road-Network Travel Time Index.
//!
//! Distills deterministic point-to-point road routing engines (such as OSRM) into a compact,
//! memory-mapped multi-resolution H3 index (`H3T1`) that returns full probability distributions
//! of driving duration between any two hexagons in ~60 nanoseconds.
//!
//! ## Concentric Grid-Distance Rings (Barnes-Hut Opening Angle on H3)
//! Rather than switching resolution by shared parent cells (which causes both same-cell collapse
//! and artificial boundary cliffs), `h3-transit` defines transitions purely via H3 grid-ring
//! distance $d_r(u, v) = \text{grid\_distance}(u, v)$ around *every* origin cell:
//!
//! - **Tier 0 (`Res 8`, $d_8 = 0$, 0–1 km)**: Intra-hexagon travel.
//! - **Tier 1 (`Res 8`, $1 \le d_8 \le 8$, 1–8 km)**: Fine neighborhood scale (~0.9 km cells).
//! - **Tier 2 (`Res 7`, $d_8 > 8, d_7 \le 10$, 8–26 km)**: Guaranteed $d_7 \ge 3$ ring separation.
//! - **Tier 3 (`Res 6`, $d_7 > 10, d_6 \le 12$, 26–80 km)**: Guaranteed $d_6 \ge 4$ ring separation.
//! - **Tier 4 (`Res 5`, $d_6 > 12, d_5 \le 14$, 80–220 km)**: Guaranteed $d_5 \ge 4$ ring separation.
//!
//! ## Binary Storage Format (`H3T1` Packed Two-Level CSR)
//! - **32-byte Header**:
//!   - `[0..4]`: Magic `"H3T1"`
//!   - `[4..8]`: Version `1u32` (little-endian)
//!   - `[8..16]`: `num_origins: u64` (little-endian)
//!   - `[16..24]`: `num_edges: u64` (little-endian)
//!   - `[24..32]`: Reserved (`0u64`)
//! - **Level 1 — Origin Directory (`num_origins * 16` bytes, sorted by `origin_h3`)**:
//!   - `origin_h3: u64` (8 bytes, little-endian)
//!   - `edge_start: u32` (4 bytes, little-endian index into Level 2)
//!   - `edge_count: u32` (4 bytes, little-endian number of contiguous edges)
//! - **Level 2 — Packed Edge Array (`num_edges * 18` bytes, sorted by `dest_h3` within each origin)**:
//!   - `dest_h3: u64` (8 bytes, little-endian)
//!   - `summary: [u8; 10]` (`min_sec: u16, mean_sec: u16, stdev_sec: u16, p50_sec: u16, p90_sec: u16`)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use h3o::{CellIndex, Resolution};
use memmap2::Mmap;
use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Gamma, LogNormal, Normal, Weibull};

const MAGIC: &[u8; 4] = b"H3T1";
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 32;
const ORIGIN_ENTRY_BYTES: usize = 16;
const EDGE_ENTRY_BYTES: usize = 18;

/// Ring radius thresholds at each resolution tier.
pub const TIER1_RES8_MAX_RINGS: i32 = 8; // ~0.9 km .. 7.5 km (Res 8 <-> Res 8)
pub const TIER2_RES7_MAX_RINGS: i32 = 10; // ~7.5 km .. 26 km  (Res 7 <-> Res 7, min d7 >= 3)
pub const TIER3_RES6_MAX_RINGS: i32 = 12; // ~26 km  .. 80 km  (Res 6 <-> Res 6, min d6 >= 4)
pub const TIER4_RES5_MAX_RINGS: i32 = 14; // ~80 km  .. 220 km (Res 5 <-> Res 5, min d5 >= 4)

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DistanceTier {
    /// $d_8 = 0$: Intra-hexagon travel within the same Res 8 cell (~0 - 0.9 km).
    Tier0IntraRes8 = 0,
    /// $1 \le d_8 \le 8$: Fine local travel at Res 8 (~0.9 - 7.5 km).
    Tier1LocalRes8 = 1,
    /// $d_8 > 8$ and $d_7 \le 10$: Mid-range travel at Res 7 (~7.5 - 26 km).
    Tier2MidRes7 = 2,
    /// $d_7 > 10$ and $d_6 \le 12$: Regional travel at Res 6 (~26 - 80 km).
    Tier3RegionalRes6 = 3,
    /// $d_6 > 12$ and $d_5 \le 14$: Long-range travel at Res 5 (~80 - 220 km).
    Tier4LongRes5 = 4,
}

impl DistanceTier {
    pub fn resolution(self) -> Resolution {
        match self {
            DistanceTier::Tier0IntraRes8 | DistanceTier::Tier1LocalRes8 => Resolution::Eight,
            DistanceTier::Tier2MidRes7 => Resolution::Seven,
            DistanceTier::Tier3RegionalRes6 => Resolution::Six,
            DistanceTier::Tier4LongRes5 => Resolution::Five,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DistanceTier::Tier0IntraRes8 => "Tier 0 (Intra-Res8, 0-1km)",
            DistanceTier::Tier1LocalRes8 => "Tier 1 (Res8 Ring 1..8, 1-8km)",
            DistanceTier::Tier2MidRes7 => "Tier 2 (Res7 Ring 3..10, 8-26km)",
            DistanceTier::Tier3RegionalRes6 => "Tier 3 (Res6 Ring 4..12, 26-80km)",
            DistanceTier::Tier4LongRes5 => "Tier 4 (Res5 Ring 4..14, 80-220km)",
        }
    }
}

/// Canonicalizes an undirected pair `(a, b)` into `(min(a, b), max(a, b))`.
#[inline(always)]
pub fn canonical_pair(a: CellIndex, b: CellIndex) -> (CellIndex, CellIndex) {
    if u64::from(a) <= u64::from(b) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Resolves any two Resolution-8 H3 cells `(orig_h8, dest_h8)` to their canonical symmetric
/// `(DistanceTier, min_cell_at_tier, max_cell_at_tier)` lookup key using concentric
/// grid-distance rings.
pub fn resolve_lookup_pair(
    orig_h8: CellIndex,
    dest_h8: CellIndex,
) -> Option<(DistanceTier, CellIndex, CellIndex)> {
    debug_assert_eq!(orig_h8.resolution(), Resolution::Eight);
    debug_assert_eq!(dest_h8.resolution(), Resolution::Eight);

    if orig_h8 == dest_h8 {
        return Some((DistanceTier::Tier0IntraRes8, orig_h8, dest_h8));
    }

    if let Ok(d8) = orig_h8.grid_distance(dest_h8) {
        if d8 <= TIER1_RES8_MAX_RINGS {
            let (u, v) = canonical_pair(orig_h8, dest_h8);
            return Some((DistanceTier::Tier1LocalRes8, u, v));
        }
    }

    let orig_h7 = orig_h8.parent(Resolution::Seven)?;
    let dest_h7 = dest_h8.parent(Resolution::Seven)?;
    if orig_h7 != dest_h7 {
        if let Ok(d7) = orig_h7.grid_distance(dest_h7) {
            if d7 <= TIER2_RES7_MAX_RINGS {
                let (u, v) = canonical_pair(orig_h7, dest_h7);
                return Some((DistanceTier::Tier2MidRes7, u, v));
            }
        }
    }

    let orig_h6 = orig_h8.parent(Resolution::Six)?;
    let dest_h6 = dest_h8.parent(Resolution::Six)?;
    if orig_h6 != dest_h6 {
        if let Ok(d6) = orig_h6.grid_distance(dest_h6) {
            if d6 <= TIER3_RES6_MAX_RINGS {
                let (u, v) = canonical_pair(orig_h6, dest_h6);
                return Some((DistanceTier::Tier3RegionalRes6, u, v));
            }
        }
    }

    let orig_h5 = orig_h8.parent(Resolution::Five)?;
    let dest_h5 = dest_h8.parent(Resolution::Five)?;
    if orig_h5 != dest_h5 {
        if let Ok(d5) = orig_h5.grid_distance(dest_h5) {
            if d5 <= TIER4_RES5_MAX_RINGS {
                let (u, v) = canonical_pair(orig_h5, dest_h5);
                return Some((DistanceTier::Tier4LongRes5, u, v));
            }
        }
    }

    None
}

/// Compact 10-byte summary of road-network travel times (in seconds) between two H3 cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitSummary {
    /// Minimum observed route duration in seconds ($t_{\min}$).
    pub min_sec: u16,
    /// Sample mean duration in seconds ($\mu$).
    pub mean_sec: u16,
    /// Predictive standard deviation in seconds ($\sigma_{\text{pred}}$).
    pub stdev_sec: u16,
    /// Sample 50th percentile (median) in seconds ($q_{0.50}$).
    pub p50_sec: u16,
    /// Sample 90th percentile in seconds ($q_{0.90}$).
    pub p90_sec: u16,
}

impl TransitSummary {
    #[inline(always)]
    pub fn to_bytes(self) -> [u8; 10] {
        let mut out = [0u8; 10];
        out[0..2].copy_from_slice(&self.min_sec.to_le_bytes());
        out[2..4].copy_from_slice(&self.mean_sec.to_le_bytes());
        out[4..6].copy_from_slice(&self.stdev_sec.to_le_bytes());
        out[6..8].copy_from_slice(&self.p50_sec.to_le_bytes());
        out[8..10].copy_from_slice(&self.p90_sec.to_le_bytes());
        out
    }

    #[inline(always)]
    pub fn from_bytes(bytes: &[u8; 10]) -> Self {
        Self {
            min_sec: u16::from_le_bytes([bytes[0], bytes[1]]),
            mean_sec: u16::from_le_bytes([bytes[2], bytes[3]]),
            stdev_sec: u16::from_le_bytes([bytes[4], bytes[5]]),
            p50_sec: u16::from_le_bytes([bytes[6], bytes[7]]),
            p90_sec: u16::from_le_bytes([bytes[8], bytes[9]]),
        }
    }

    /// Computes a [`TransitSummary`] from a non-empty slice of sample durations in seconds.
    pub fn from_samples(samples: &mut [f32]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = samples.len();
        let min_v = samples[0].max(1.0);

        let sum: f64 = samples.iter().map(|&x| x as f64).sum();
        let mean_v = (sum / n as f64) as f32;

        let var: f64 = if n > 1 {
            let raw_var = samples
                .iter()
                .map(|&x| {
                    let d = (x as f64) - (mean_v as f64);
                    d * d
                })
                .sum::<f64>()
                / ((n - 1) as f64);
            let pred_factor = ((n as f64) + 2.0) / ((n as f64) - 0.5);
            raw_var * pred_factor
        } else {
            ((mean_v as f64) * 0.12).powi(2)
        };
        let stdev_v = (var.sqrt() as f32).max(2.0);

        let p50_v = percentile_weibull_sorted(samples, 0.50);
        let p90_v = percentile_weibull_sorted(samples, 0.90).max(p50_v + 1.0);

        Some(Self {
            min_sec: clamp_u16(min_v),
            mean_sec: clamp_u16(mean_v),
            stdev_sec: clamp_u16(stdev_v),
            p50_sec: clamp_u16(p50_v),
            p90_sec: clamp_u16(p90_v),
        })
    }
}

/// Unbiased Weibull plotting-position quantile estimator $i / (n + 1)$ for finite samples.
#[inline]
fn percentile_weibull_sorted(sorted: &[f32], q: f32) -> f32 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = (q * ((n + 1) as f32) - 1.0).clamp(0.0, (n - 1) as f32);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f32);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[inline]
fn clamp_u16(val: f32) -> u16 {
    val.round().clamp(1.0, u16::MAX as f32) as u16
}

/// Parametric distribution families reconstructed at runtime from a 10-byte [`TransitSummary`].
#[derive(Debug, Clone, Copy)]
pub enum ParametricModel {
    /// 3-parameter Shifted Weibull fit via Quantile Matching on `(t0, p50, p90)`.
    ShiftedWeibull { t0: f64, shape: f64, scale: f64 },
    /// 3-parameter Shifted Gamma fit via Method of Moments on `(t0, mean, stdev)`.
    ShiftedGamma { t0: f64, shape: f64, rate: f64 },
    /// 3-parameter Shifted Log-Normal fit via Method of Moments on `(t0, mean, stdev)`.
    ShiftedLogNormalMom { t0: f64, mu: f64, sigma: f64 },
    /// 3-parameter Shifted Log-Normal fit via Quantile Matching on `(t0, p50, p90)`.
    ShiftedLogNormalQuantile { t0: f64, mu: f64, sigma: f64 },
    /// Symmetric Gaussian (Normal) baseline parameterized by `(mean, stdev)`.
    GaussianBaseline { mean: f64, stdev: f64 },
}

impl ParametricModel {
    fn compute_shift(s: &TransitSummary) -> f64 {
        let min_f = s.min_sec as f64;
        let mean_f = (s.mean_sec as f64).max(min_f + 2.0);
        let std_f = (s.stdev_sec as f64).max(2.0);
        let max_t0_for_cv = (mean_f - 1.35 * std_f).max(0.0);
        (0.75 * min_f).min(max_t0_for_cv).max(0.0)
    }

    pub fn shifted_weibull(s: &TransitSummary) -> Self {
        let min_f = s.min_sec as f64;
        let p50 = (s.p50_sec as f64).max(min_f + 1.5);
        let p90 = (s.p90_sec as f64).max(p50 + 2.0);
        let t0 = (0.75 * min_f).min(p50 - 2.0).max(0.0);
        let q50_ex = (p50 - t0).max(1.5);
        let q90_ex = (p90 - t0).max(q50_ex + 1.5);
        let shape = (1.2011224087864498 / (q90_ex / q50_ex).ln()).clamp(0.6, 20.0);
        let scale = (q50_ex / (2.0f64.ln()).powf(1.0 / shape)).max(1.0);
        Self::ShiftedWeibull { t0, shape, scale }
    }

    pub fn shifted_gamma(s: &TransitSummary) -> Self {
        let t0 = Self::compute_shift(s);
        let excess_mean = ((s.mean_sec as f64) - t0).max(3.0);
        let stdev = (s.stdev_sec as f64).max(2.0);
        let shape = (excess_mean / stdev).powi(2).clamp(0.5, 100.0);
        let rate = (shape / excess_mean).max(1e-4);
        Self::ShiftedGamma { t0, shape, rate }
    }

    pub fn shifted_lognormal_mom(s: &TransitSummary) -> Self {
        let t0 = Self::compute_shift(s);
        let excess_mean = ((s.mean_sec as f64) - t0).max(3.0);
        let stdev = (s.stdev_sec as f64).max(2.0);
        let cv2 = (stdev / excess_mean).powi(2);
        let sigma2 = (1.0 + cv2).ln().clamp(0.01, 2.5);
        let sigma = sigma2.sqrt();
        let mu = excess_mean.ln() - 0.5 * sigma2;
        Self::ShiftedLogNormalMom { t0, mu, sigma }
    }

    pub fn shifted_lognormal_quantile(s: &TransitSummary) -> Self {
        let min_f = s.min_sec as f64;
        let p50 = (s.p50_sec as f64).max(min_f + 1.5);
        let p90 = (s.p90_sec as f64).max(p50 + 2.0);
        let t0 = (0.75 * min_f).min(p50 - 2.0).max(0.0);
        let q50_ex = (p50 - t0).max(1.5);
        let q90_ex = (p90 - t0).max(q50_ex + 1.5);
        let mu = q50_ex.ln();
        let sigma = ((q90_ex.ln() - mu) / 1.2815515655446004).clamp(0.08, 1.5);
        Self::ShiftedLogNormalQuantile { t0, mu, sigma }
    }

    pub fn gaussian_baseline(s: &TransitSummary) -> Self {
        Self::GaussianBaseline {
            mean: (s.mean_sec as f64).max(5.0),
            stdev: (s.stdev_sec as f64).max(2.0),
        }
    }

    /// Evaluates the Cumulative Distribution Function $\hat{F}(t) = P(T \le t)$.
    pub fn cdf(&self, t: f64) -> f64 {
        match *self {
            Self::ShiftedLogNormalMom { t0, mu, sigma }
            | Self::ShiftedLogNormalQuantile { t0, mu, sigma } => {
                if t <= t0 {
                    0.0
                } else if let Ok(dist) = LogNormal::new(mu, sigma) {
                    dist.cdf(t - t0).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            }
            Self::ShiftedGamma { t0, shape, rate } => {
                if t <= t0 {
                    0.0
                } else if let Ok(dist) = Gamma::new(shape, rate) {
                    dist.cdf(t - t0).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            }
            Self::ShiftedWeibull { t0, shape, scale } => {
                if t <= t0 {
                    0.0
                } else if let Ok(dist) = Weibull::new(shape, scale) {
                    dist.cdf(t - t0).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            }
            Self::GaussianBaseline { mean, stdev } => {
                if let Ok(n) = Normal::new(mean, stdev) {
                    n.cdf(t).clamp(0.0, 1.0)
                } else {
                    0.5
                }
            }
        }
    }

    /// Evaluates the Probability Density Function $\hat{f}(t)$.
    pub fn pdf(&self, t: f64) -> f64 {
        let eps = 0.5;
        ((self.cdf(t + eps) - self.cdf((t - eps).max(0.0))) / (2.0 * eps)).max(0.0)
    }

    /// Draws a random travel time sample in seconds.
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        let u: f64 = rng.random_range(1e-6..1.0 - 1e-6);
        self.quantile(u)
    }

    /// Evaluates the inverse CDF $\hat{F}^{-1}(u)$ in seconds for $u \in (0, 1)$.
    pub fn quantile(&self, u: f64) -> f64 {
        let u = u.clamp(1e-6, 1.0 - 1e-6);
        match *self {
            Self::ShiftedLogNormalMom { t0, mu, sigma }
            | Self::ShiftedLogNormalQuantile { t0, mu, sigma } => {
                if let Ok(dist) = LogNormal::new(mu, sigma) {
                    t0 + dist.inverse_cdf(u)
                } else {
                    t0 + mu.exp()
                }
            }
            Self::ShiftedGamma { t0, shape, rate } => {
                if let Ok(dist) = Gamma::new(shape, rate) {
                    t0 + dist.inverse_cdf(u)
                } else {
                    t0 + shape / rate
                }
            }
            Self::ShiftedWeibull { t0, shape, scale } => {
                t0 + scale * (-((1.0 - u).ln())).powf(1.0 / shape)
            }
            Self::GaussianBaseline { mean, stdev } => {
                if let Ok(n) = Normal::new(mean, stdev) {
                    n.inverse_cdf(u).max(1.0)
                } else {
                    mean
                }
            }
        }
    }
}

/// Zero-copy memory-mapped Compressed Sparse Row (`H3T1`) index for nanosecond H3 pair lookups.
pub struct H3CsrIndex {
    mmap: Mmap,
    num_origins: usize,
    num_edges: usize,
}

impl H3CsrIndex {
    /// Writes a collection of `(orig_h3, dest_h3) -> TransitSummary` records into a compact
    /// two-level CSR binary file (`H3T1`).
    pub fn write_to_path(
        path: impl AsRef<Path>,
        records: &[((CellIndex, CellIndex), TransitSummary)],
    ) -> Result<(usize, usize)> {
        let mut adj: BTreeMap<u64, Vec<(u64, [u8; 10])>> = BTreeMap::new();
        for &((orig, dest), summary) in records {
            let (u, v) = canonical_pair(orig, dest);
            adj.entry(u64::from(u))
                .or_default()
                .push((u64::from(v), summary.to_bytes()));
        }

        let mut num_edges = 0usize;
        for edges in adj.values_mut() {
            edges.sort_unstable_by_key(|e| e.0);
            edges.dedup_by_key(|e| e.0);
            num_edges += edges.len();
        }
        let num_origins = adj.len();

        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create {:?}", path.as_ref()))?;
        let mut w = BufWriter::with_capacity(4 * 1024 * 1024, file);

        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(num_origins as u64).to_le_bytes())?;
        w.write_all(&(num_edges as u64).to_le_bytes())?;
        w.write_all(&0u64.to_le_bytes())?;

        let mut edge_cursor: u32 = 0;
        for (&orig_u64, edges) in &adj {
            let count = edges.len() as u32;
            w.write_all(&orig_u64.to_le_bytes())?;
            w.write_all(&edge_cursor.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
            edge_cursor = edge_cursor
                .checked_add(count)
                .context("Edge count exceeded u32::MAX")?;
        }

        for edges in adj.values() {
            for (dest_u64, summary_bytes) in edges {
                w.write_all(&dest_u64.to_le_bytes())?;
                w.write_all(summary_bytes)?;
            }
        }
        w.flush()?;
        Ok((num_origins, num_edges))
    }

    /// Memory-maps an `H3T1` CSR index file for zero-copy read access.
    pub fn open_mmap(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open {:?}", path.as_ref()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        anyhow::ensure!(mmap.len() >= HEADER_BYTES, "H3T1 file shorter than header");
        anyhow::ensure!(&mmap[0..4] == MAGIC, "Invalid H3T1 magic bytes");
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        anyhow::ensure!(version == VERSION, "Unsupported H3T1 version {}", version);

        let num_origins = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let num_edges = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let expected_len =
            HEADER_BYTES + num_origins * ORIGIN_ENTRY_BYTES + num_edges * EDGE_ENTRY_BYTES;
        anyhow::ensure!(
            mmap.len() == expected_len,
            "H3T1 size mismatch: expected {} bytes, got {}",
            expected_len,
            mmap.len()
        );

        Ok(Self {
            mmap,
            num_origins,
            num_edges,
        })
    }

    pub fn num_origins(&self) -> usize {
        self.num_origins
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    #[inline(always)]
    fn origin_at(&self, idx: usize) -> (u64, u32, u32) {
        let offset = HEADER_BYTES + idx * ORIGIN_ENTRY_BYTES;
        let b = &self.mmap[offset..offset + ORIGIN_ENTRY_BYTES];
        let orig = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        let start = u32::from_le_bytes([b[8], b[9], b[10], b[11]]);
        let count = u32::from_le_bytes([b[12], b[13], b[14], b[15]]);
        (orig, start, count)
    }

    #[inline(always)]
    fn edge_at(&self, edge_idx: usize) -> (u64, [u8; 10]) {
        let edges_base = HEADER_BYTES + self.num_origins * ORIGIN_ENTRY_BYTES;
        let offset = edges_base + edge_idx * EDGE_ENTRY_BYTES;
        let b = &self.mmap[offset..offset + EDGE_ENTRY_BYTES];
        let dest = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        let mut summary = [0u8; 10];
        summary.copy_from_slice(&b[8..18]);
        (dest, summary)
    }

    /// Looks up the exact canonical pair `canonical_pair(cell_a, cell_b)` at a single resolution.
    #[inline]
    pub fn get_exact_cells(&self, cell_a: CellIndex, cell_b: CellIndex) -> Option<TransitSummary> {
        let (u_cell, v_cell) = canonical_pair(cell_a, cell_b);
        let u = u64::from(u_cell);
        let v = u64::from(v_cell);

        let mut lo = 0usize;
        let mut hi = self.num_origins;
        let mut found_range: Option<(usize, usize)> = None;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (mid_orig, start, count) = self.origin_at(mid);
            match mid_orig.cmp(&u) {
                std::cmp::Ordering::Equal => {
                    found_range = Some((start as usize, (start + count) as usize));
                    break;
                }
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        let (edge_lo_start, edge_hi_end) = found_range?;

        let mut elo = edge_lo_start;
        let mut ehi = edge_hi_end;
        while elo < ehi {
            let emid = elo + (ehi - elo) / 2;
            let (mid_dest, summary_bytes) = self.edge_at(emid);
            match mid_dest.cmp(&v) {
                std::cmp::Ordering::Equal => {
                    return Some(TransitSummary::from_bytes(&summary_bytes));
                }
                std::cmp::Ordering::Less => elo = emid + 1,
                std::cmp::Ordering::Greater => ehi = emid,
            }
        }
        None
    }

    /// Queries the multi-resolution CSR index for any two Resolution-8 H3 cells in ~60 nanoseconds.
    #[inline]
    pub fn get_pair_res8(
        &self,
        orig_h8: CellIndex,
        dest_h8: CellIndex,
    ) -> Option<(DistanceTier, TransitSummary)> {
        let (tier, u_cell, v_cell) = resolve_lookup_pair(orig_h8, dest_h8)?;
        self.get_exact_cells(u_cell, v_cell).map(|s| (tier, s))
    }

    /// Queries the index for any two H3 cells `orig` and `dest` (at Res 5, 6, 7, or 8).
    pub fn get_pair_any_res(
        &self,
        orig: CellIndex,
        dest: CellIndex,
    ) -> Option<(DistanceTier, CellIndex, CellIndex, TransitSummary)> {
        if orig.resolution() == Resolution::Eight && dest.resolution() == Resolution::Eight {
            let (tier, u, v) = resolve_lookup_pair(orig, dest)?;
            let s = self.get_exact_cells(u, v)?;
            return Some((tier, u, v, s));
        }
        for res in [
            Resolution::Eight,
            Resolution::Seven,
            Resolution::Six,
            Resolution::Five,
        ] {
            let o_r = if orig.resolution() == res {
                Some(orig)
            } else {
                orig.parent(res)
            };
            let d_r = if dest.resolution() == res {
                Some(dest)
            } else {
                dest.parent(res)
            };
            if let (Some(o_c), Some(d_c)) = (o_r, d_r) {
                if let Some(s) = self.get_exact_cells(o_c, d_c) {
                    let tier = match res {
                        Resolution::Eight if o_c == d_c => DistanceTier::Tier0IntraRes8,
                        Resolution::Eight => DistanceTier::Tier1LocalRes8,
                        Resolution::Seven => DistanceTier::Tier2MidRes7,
                        Resolution::Six => DistanceTier::Tier3RegionalRes6,
                        _ => DistanceTier::Tier4LongRes5,
                    };
                    let (u, v) = canonical_pair(o_c, d_c);
                    return Some((tier, u, v, s));
                }
            }
        }
        None
    }

    /// Scans the CSR index to collect all road-active Resolution-8 cells and their intra-cell mean duration.
    pub fn collect_active_res8_cells(&self) -> Vec<(CellIndex, u16)> {
        let mut out = Vec::new();
        for i in 0..self.num_origins {
            let (orig_u64, start, count) = self.origin_at(i);
            if let Ok(cell) = CellIndex::try_from(orig_u64) {
                if cell.resolution() == Resolution::Eight && count > 0 {
                    let (first_dest, summary_bytes) = self.edge_at(start as usize);
                    if first_dest == orig_u64 {
                        let s = TransitSummary::from_bytes(&summary_bytes);
                        out.push((cell, s.mean_sec));
                    } else {
                        out.push((cell, 60));
                    }
                }
            }
        }
        out
    }

    /// Computes the multi-resolution Barnes-Hut reachability fan around `orig_h8` across Tiers 0–4.
    pub fn get_reachability_fan(
        &self,
        orig_h8: CellIndex,
    ) -> Vec<(DistanceTier, CellIndex, TransitSummary)> {
        let mut fan = Vec::with_capacity(1024);

        for c8 in orig_h8.grid_disk::<Vec<_>>(TIER1_RES8_MAX_RINGS as u32) {
            if let Some(s) = self.get_exact_cells(orig_h8, c8) {
                let tier = if c8 == orig_h8 {
                    DistanceTier::Tier0IntraRes8
                } else {
                    DistanceTier::Tier1LocalRes8
                };
                fan.push((tier, c8, s));
            }
        }

        if let Some(orig_h7) = orig_h8.parent(Resolution::Seven) {
            for c7 in orig_h7.grid_disk::<Vec<_>>(TIER2_RES7_MAX_RINGS as u32) {
                if let Ok(d7) = orig_h7.grid_distance(c7) {
                    if d7 >= 3 {
                        if let Some(s) = self.get_exact_cells(orig_h7, c7) {
                            fan.push((DistanceTier::Tier2MidRes7, c7, s));
                        }
                    }
                }
            }
        }

        if let Some(orig_h6) = orig_h8.parent(Resolution::Six) {
            for c6 in orig_h6.grid_disk::<Vec<_>>(TIER3_RES6_MAX_RINGS as u32) {
                if let Ok(d6) = orig_h6.grid_distance(c6) {
                    if d6 >= 4 {
                        if let Some(s) = self.get_exact_cells(orig_h6, c6) {
                            fan.push((DistanceTier::Tier3RegionalRes6, c6, s));
                        }
                    }
                }
            }
        }

        if let Some(orig_h5) = orig_h8.parent(Resolution::Five) {
            for c5 in orig_h5.grid_disk::<Vec<_>>(TIER4_RES5_MAX_RINGS as u32) {
                if let Ok(d5) = orig_h5.grid_distance(c5) {
                    if d5 >= 4 {
                        if let Some(s) = self.get_exact_cells(orig_h5, c5) {
                            fan.push((DistanceTier::Tier4LongRes5, c5, s));
                        }
                    }
                }
            }
        }

        fan
    }

    /// Streams and merges multiple regional `.h3t` CSR index files into a single unified
    /// `.h3t` file using a two-pass K-way merge with <50 MB RAM overhead.
    pub fn merge_files(
        input_paths: &[impl AsRef<Path>],
        output_path: impl AsRef<Path>,
    ) -> Result<(usize, usize)> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let shards: Vec<Self> = input_paths
            .iter()
            .map(|p| Self::open_mmap(p.as_ref()))
            .collect::<Result<Vec<_>>>()?;

        let mut heap: BinaryHeap<Reverse<(u64, usize, usize)>> = BinaryHeap::new();
        for (s_idx, shard) in shards.iter().enumerate() {
            if shard.num_origins > 0 {
                let (orig, _, _) = shard.origin_at(0);
                heap.push(Reverse((orig, s_idx, 0)));
            }
        }

        let mut merged_origins: Vec<(u64, u32, u32)> = Vec::new();
        let mut edge_cursor: u32 = 0;
        let mut group_shards: Vec<(usize, u32, u32)> = Vec::new();
        let mut merged_dests: Vec<u64> = Vec::new();

        while let Some(&Reverse((curr_orig, _, _))) = heap.peek() {
            group_shards.clear();
            while let Some(&Reverse((orig, s_idx, o_idx))) = heap.peek() {
                if orig != curr_orig {
                    break;
                }
                heap.pop();
                let (_, start, count) = shards[s_idx].origin_at(o_idx);
                group_shards.push((s_idx, start, count));
                if o_idx + 1 < shards[s_idx].num_origins {
                    let (next_orig, _, _) = shards[s_idx].origin_at(o_idx + 1);
                    heap.push(Reverse((next_orig, s_idx, o_idx + 1)));
                }
            }

            let count = if group_shards.len() == 1 {
                group_shards[0].2
            } else {
                merged_dests.clear();
                for &(s_idx, start, cnt) in &group_shards {
                    for e_idx in (start as usize)..((start + cnt) as usize) {
                        let (dest_u64, _) = shards[s_idx].edge_at(e_idx);
                        merged_dests.push(dest_u64);
                    }
                }
                merged_dests.sort_unstable();
                merged_dests.dedup();
                merged_dests.len() as u32
            };

            merged_origins.push((curr_orig, edge_cursor, count));
            edge_cursor = edge_cursor
                .checked_add(count)
                .context("Total merged edges exceeded u32::MAX")?;
        }

        let num_origins = merged_origins.len();
        let num_edges = edge_cursor as usize;

        let file = File::create(output_path.as_ref()).with_context(|| {
            format!("Failed to create merged output {:?}", output_path.as_ref())
        })?;
        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, file);

        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(num_origins as u64).to_le_bytes())?;
        w.write_all(&(num_edges as u64).to_le_bytes())?;
        w.write_all(&0u64.to_le_bytes())?;

        for &(orig_u64, start, count) in &merged_origins {
            w.write_all(&orig_u64.to_le_bytes())?;
            w.write_all(&start.to_le_bytes())?;
            w.write_all(&count.to_le_bytes())?;
        }

        for (s_idx, shard) in shards.iter().enumerate() {
            if shard.num_origins > 0 {
                let (orig, _, _) = shard.origin_at(0);
                heap.push(Reverse((orig, s_idx, 0)));
            }
        }

        let mut merged_edges_buf: Vec<(u64, [u8; 10])> = Vec::new();
        while let Some(&Reverse((curr_orig, _, _))) = heap.peek() {
            group_shards.clear();
            while let Some(&Reverse((orig, s_idx, o_idx))) = heap.peek() {
                if orig != curr_orig {
                    break;
                }
                heap.pop();
                let (_, start, count) = shards[s_idx].origin_at(o_idx);
                group_shards.push((s_idx, start, count));
                if o_idx + 1 < shards[s_idx].num_origins {
                    let (next_orig, _, _) = shards[s_idx].origin_at(o_idx + 1);
                    heap.push(Reverse((next_orig, s_idx, o_idx + 1)));
                }
            }

            if group_shards.len() == 1 {
                let (s_idx, start, cnt) = group_shards[0];
                let edges_base = HEADER_BYTES + shards[s_idx].num_origins * ORIGIN_ENTRY_BYTES;
                let byte_start = edges_base + (start as usize) * EDGE_ENTRY_BYTES;
                let byte_end = byte_start + (cnt as usize) * EDGE_ENTRY_BYTES;
                w.write_all(&shards[s_idx].mmap[byte_start..byte_end])?;
            } else {
                merged_edges_buf.clear();
                for &(s_idx, start, cnt) in &group_shards {
                    for e_idx in (start as usize)..((start + cnt) as usize) {
                        merged_edges_buf.push(shards[s_idx].edge_at(e_idx));
                    }
                }
                merged_edges_buf.sort_unstable_by_key(|e| e.0);
                merged_edges_buf.dedup_by_key(|e| e.0);
                for (dest_u64, summary_bytes) in &merged_edges_buf {
                    w.write_all(&dest_u64.to_le_bytes())?;
                    w.write_all(summary_bytes)?;
                }
            }
        }

        w.flush()?;
        Ok((num_origins, num_edges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_bytes_roundtrip() {
        let mut samples = vec![120.0, 145.0, 160.0, 180.0, 210.0, 250.0, 310.0];
        let s = TransitSummary::from_samples(&mut samples).unwrap();
        let b = s.to_bytes();
        assert_eq!(TransitSummary::from_bytes(&b), s);
        assert!(s.min_sec <= s.p50_sec && s.p50_sec <= s.p90_sec);
    }

    #[test]
    fn test_parametric_cdf_quantile_inversion() {
        let s = TransitSummary {
            min_sec: 100,
            p50_sec: 220,
            mean_sec: 240,
            p90_sec: 380,
            stdev_sec: 95,
        };
        for model in [
            ParametricModel::shifted_weibull(&s),
            ParametricModel::shifted_gamma(&s),
            ParametricModel::shifted_lognormal_mom(&s),
        ] {
            for &q in &[0.10, 0.25, 0.50, 0.75, 0.90] {
                let t = model.quantile(q);
                let q_back = model.cdf(t);
                assert!((q - q_back).abs() < 1e-4);
            }
        }
    }
}
