# `h3-transit`

**Multi-Resolution H3 Stochastic Road-Network Travel Time Index & PIT Calibration Suite**

`h3-transit` distills deterministic [OpenStreetMap road routing graphs (OSRM)](https://project-osrm.org/) into a compact, zero-copy memory-mapped **Compressed Sparse Row (`H3T1`)** spatial index over Uber's hierarchical **[H3 hexagonal grid](https://h3geo.org/)**. Instead of returning a single point-to-point scalar duration, `h3-transit` returns the **probability distribution of driving travel times $T_{A \to B}$** between any two zones in **tens of nanoseconds**—capturing the geometric and road-network variance across all drivable streets within each hexagon.

---

## Key Highlights

- **Barnes-Hut Concentric H3 Rings (Zero Boundary Cliffs)**: Bounded opening angle $\theta = \text{diam}(H_r) / \text{dist}(A, B) \le 0.25$ across all distance scales using H3 grid-distance rings (`Res 8` fine local $\to$ `Res 7` mid-range $\to$ `Res 6` regional $\to$ `Res 5` long-range), eliminating both same-parent self-loop collapse and parent-boundary discretization jumps.
- **Fast Memory-Mapped CSR Lookup (`H3T1`)**: Stores canonical undirected pairs `(min(A, B), max(A, B))` in a two-level static Compressed Sparse Row binary file (`18 bytes/edge`). All of New Jersey (`4,271,969` canonical pairs across `45,708` active origins) fits in **74.03 MB** (`7.0x` smaller and `236x` faster than a B+Tree).
- **Distribution-Agnostic 10-Byte Summary Record**: Each edge stores `(min_sec, mean_sec, stdev_sec, p50_sec, p90_sec)` as $5 \times \texttt{u16}$ (`10 bytes`), allowing runtime reconstruction of **3-Parameter Shifted Weibull**, **Shifted Gamma**, or **Shifted Log-Normal** distributions via either Method of Moments or Quantile Matching.
- **Built-In Probability Integral Transform (PIT) Backtester**: Validates empirical quantile coverage ($P(u \le 10\%), P(u \le 25\%), P(u \le 50\%), P(u \le 75\%), P(u \le 90\%)$) and Kolmogorov-Smirnov ($D_{\text{KS}}$) distance against holdout random road-vertex pairs queried on live OSRM.
- **Low-Memory Nationwide Iterative Builder (`build-us`)**: Top-down recursive `.osm.pbf` bisection + streaming K-way CSR merge builds the entire 50-state United States on a 16 GB machine with zero swap.

---

## 1. Resolving the Hierarchical "Boundary Cliff" & "Same-Parent Collapse"

A naive H3 hierarchy (*"use Res 8 if two cells share a Res 5 parent, otherwise use Res 5"*) fails in two ways:
1. **Same-Parent Collapse**: A Res 5 cell is ~17 km wide. Two points 14 km apart inside the *same* Res 5 cell collapse to `parent(A) == parent(B)`.
2. **Boundary Cliffs**: Two Res 8 cells immediately adjacent (0.8 km apart) across the border of two different Res 5 cells suffer an artificial 17 km discretization jump.

`h3-transit` defines resolution transitions purely via **H3 grid-ring distance $d_r(u, v) = \text{grid\_distance}(u, v)$** around *every* origin cell:

| Tier | Resolution | Hex Width | Ring Condition | Physical Distance Band | Minimum Separation at Handoff |
|---|---|---|---|---|---|
| **Tier 0 (Intra-Hex)** | Res 8 $\leftrightarrow$ Res 8 | ~0.9 km | $d_8(A_8, B_8) = 0$ | $0 - 0.9\text{ km}$ | Unstratified intra-cell road pairs |
| **Tier 1 (Fine Local)** | Res 8 $\leftrightarrow$ Res 8 | ~0.9 km | $1 \le d_8(A_8, B_8) \le 8$ | $0.9 - 7.5\text{ km}$ | Always distinct ($d_8 \ge 1$) |
| **Tier 2 (Mid-Range)** | Res 7 $\leftrightarrow$ Res 7 | ~2.6 km | $d_8 > 8$ and $d_7(A_7, B_7) \le 10$ | $7.5 - 26\text{ km}$ | $d_7(A_7, B_7) \ge 3$ rings apart |
| **Tier 3 (Regional)** | Res 6 $\leftrightarrow$ Res 6 | ~6.4 km | $d_7 > 10$ and $d_6(A_6, B_6) \le 12$ | $26 - 80\text{ km}$ | $d_6(A_6, B_6) \ge 4$ rings apart |
| **Tier 4 (Long-Range)** | Res 5 $\leftrightarrow$ Res 5 | ~17.0 km | $d_6 > 12$ and $d_5(A_5, B_5) \le 14$ | $80 - 220\text{ km}$ | $d_5(A_5, B_5) \ge 4$ rings apart |

---

## 2. Binary Format Specification (`H3T1` Two-Level CSR)

All pairs are stored in canonical undirected order `u = min(orig, dest), v = max(orig, dest)`:

1. **Header (`32 bytes`)**:
   - `[0..4]`: Magic `"H3T1"`
   - `[4..8]`: Version `1u32` (little-endian)
   - `[8..16]`: `num_origins: u64` (little-endian)
   - `[16..24]`: `num_edges: u64` (little-endian)
   - `[24..32]`: Reserved (`0u64`)
2. **Level 1 — Sorted Origin Directory (`16 bytes * num_origins`)**:
   - `origin_h3: u64` (8 bytes)
   - `edge_start: u32` (4 bytes — starting index in Level 2)
   - `edge_count: u32` (4 bytes — number of contiguous edges for this origin)
3. **Level 2 — Packed Contiguous Edge Array (`18 bytes * num_edges`)**:
   - `dest_h3: u64` (8 bytes, sorted within each origin's slice)
   - `summary: [u8; 10]` = 5 $\times$ `u16` in seconds (`min_sec`, `mean_sec`, `stdev_sec`, `p50_sec`, `p90_sec`)

---

## 3. Empirical Benchmarks & PIT Calibration Study (New Jersey)

Built from `us-261103.osm.pbf` on an Apple M4 Mac mini (16 GB RAM) covering New Jersey + NY/PA/DE border buffer (`10,624,020` directed road edges, `1,097,130` drivable highway vertices, `42,880` road-active Res-8 cells):

| Metric | Transactional B+Tree (`redb`) | `h3-transit` Packed CSR (`H3T1`) | Improvement |
|---|---|---|---|
| **Index File Size** | 514.50 MB | **74.03 MB** | **7.0x smaller** |
| **Write / Flush Time** | 5.00 s | **0.35 s** | **14x faster** |
| **Pair Lookup Latency** | 14,208 ns (14.2 µs) | **60 ns** (0.06 µs) | **236x faster** |

### Holdout Probability Integral Transform (PIT) Calibration ($N = 9,991$ Random Road Pairs)

2,000 random drivable road-vertex pairs per tier $(x_i, y_i) \to (x'_i, y'_i)$ were drawn across New Jersey and evaluated against exact point-to-point OSRM `/route/v1/driving` durations $t_i^*$, measuring $u_i = \hat{F}_i(t_i^*)$:

| Distribution Family | $P(u \le 10\%)$ | $P(u \le 25\%)$ | $P(u \le 50\%)$ | $P(u \le 75\%)$ | $P(u \le 90\%)$ | KS Stat ($D_{\text{KS}} \downarrow$) |
|---|---|---|---|---|---|---|
| *Ideal Calibrated Target* | *10.0%* | *25.0%* | *50.0%* | *75.0%* | *90.0%* | *0.0000* |
| **Shifted Weibull (Quantile)** | **11.0%** | **24.6%** | **49.3%** | **72.8%** | **87.2%** | **0.0356** |
| **Shifted Gamma (MoM)** | **13.3%** | **26.3%** | **47.5%** | **70.0%** | **87.4%** | **0.0523** |
| **Shifted LogNormal (MoM)** | **14.9%** | **26.9%** | **46.3%** | **68.6%** | **86.8%** | **0.0671** |
| Shifted LogNormal (Quantile) | 18.9% | 29.8% | 49.3% | 71.0% | 87.4% | 0.0942 |

---

## 4. CLI Usage

### Build a Single Region (`.osm.pbf` + running `osrm-routed` $\to$ `.h3t`)
```bash
cargo run --release -- build \
  --pbf nj-buffered.osm.pbf \
  --db nj_h3_transit.h3t \
  --osrm-url http://127.0.0.1:5050 \
  --core-bbox=-75.6,38.9,-73.85,41.36 \
  --samples-per-res8 3
```

### Run Holdout PIT Calibration Backtest Against OSRM
```bash
cargo run --release -- backtest \
  --pbf nj-buffered.osm.pbf \
  --db nj_h3_transit.h3t \
  --osrm-url http://127.0.0.1:5050 \
  --core-bbox=-75.6,38.9,-73.85,41.36 \
  --pairs-per-tier 2000
```

### Query Any Two H3 Cells in Nanoseconds
```bash
cargo run --release -- query \
  --db nj_h3_transit.h3t \
  --orig 882a100d29fffff \
  --dest 882a100d2bfffff
```

### Iteratively Build & Merge the Full United States on 16 GB RAM
```bash
cargo run --release -- build-us \
  --us-pbf us-latest.osm.pbf \
  --work-dir ./us_build \
  --output us_h3_transit.h3t \
  --max-tile-pbf-mb 550 \
  --osrm-port 5050
```

---

## 5. Rust Library Usage

```rust
use h3_transit::{H3CsrIndex, ParametricModel};
use h3o::CellIndex;

fn main() -> anyhow::Result<()> {
    let index = H3CsrIndex::open_mmap("nj_h3_transit.h3t")?;
    let orig: CellIndex = "882a100d29fffff".parse()?;
    let dest: CellIndex = "882a100d2bfffff".parse()?;

    if let Some((tier, summary)) = index.get_pair_res8(orig, dest) {
        let dist = ParametricModel::shifted_weibull(&summary);
        let mut rng = rand::rng();
        let sample_seconds = dist.sample(&mut rng);
        println!(
            "{} | p50={}s | sampled={:.1}s",
            tier.label(),
            summary.p50_sec,
            sample_seconds
        );
    }
    Ok(())
}
```

## License

MIT
