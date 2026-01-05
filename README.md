# Cloud Arbitrage Index (CAI) — Termination Risk Modeling

The Cloud Arbitrage Index (CAI) is a forward-looking risk metric designed to make **spot and preemptible cloud instances predictable and comparable across providers**. This repository implements the modeling pipeline used to estimate near-term termination risk for spot instances using historical probe data from AWS and Azure.

While spot instances offer steep cost discounts, their reliability is highly variable and difficult to reason about in practice. Provider-supplied metrics are typically coarse, backward-looking, or opaque, leaving engineers to choose between expensive on-demand capacity and unreliable spot usage. CAI addresses this gap by transforming raw probe measurements into **calibrated probability estimates of interruption over a specified time horizon**, enabling informed placement, scheduling, and multicloud arbitrage decisions.

At its core, this project converts probe runs into a discrete-time survival modeling problem. Each run is decomposed into hourly slices, enriched with leakage-safe rolling features that capture recent market behavior such as launch failures and interruption history. These slice-level outcomes are then aggregated into a forward-looking, pool-level target that estimates the probability of at least one termination event occurring within the next `H` hours for a given `(provider, region, instance_type)` pool.

The resulting model produces **interpretable, horizon-specific termination risk estimates** that can be compared consistently across regions, instance families, and cloud providers. By bridging active measurement with predictive modeling, CAI provides a practical foundation for risk-aware schedulers, deadline-sensitive workloads, and cost-optimized multicloud systems.

---

## What this code does

Given historical probe logs, the pipeline performs the following steps:

1. **Slice construction**  
   Each successfully launched run is expanded into **1-hour slices**, where each slice is labeled with whether the run terminates during that hour.

2. **Leakage-safe feature generation**  
   A unified event timeline (slice events plus launch-failure events) is constructed so that all rolling features are computed using **past-only information**.

3. **Forward-looking block-horizon target**  
   For each `(provider, region, instance_type)` pool, slice outcomes are aggregated over the next `H` hours to produce:
   - `y_block_horizon`: number of 1-hour slices in `[t, t + H)`
   - `y_block_count`: number of those slices that terminated
   - `y_block_rate`: `y_block_count / y_block_horizon`
   - `y_block_bin`: indicator for whether *any* termination occurs in the block

4. **Model training**  
   A **Poisson regression model** (HistGradientBoostingRegressor with `loss="poisson"`) is trained on `y_block_count`.

5. **Calibration**  
   Predicted counts are converted to rates and scaled using a **single global calibration factor**, chosen to minimize fractional log loss on the training set.

6. **Evaluation**  
   Performance is evaluated using:
   - Brier score (vs rate)
   - Fractional log loss
   - ROC-AUC on the “any termination in block” label
   - Reliability tables
   - Learning curves

---

## Project structure
src/  
cai_model/  
cli.py # entrypoint  
io.py # CSV loading and snapshot unpacking  
features.py # slicing and rolling features  
rolling.py # rolling-window utilities  
targets.py # block-horizon target construction  
model.py # Poisson regression model  
weights.py # sample weighting  
eval.py # metrics and reliability tables  
viz.py # learning curves  

## Requirements

- Python 3.9+
- Dependencies:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib

Install dependencies:  
pip install numpy pandas scikit-learn matplotlib

---

## Data format

The pipeline expects a CSV file with **at minimum** the following columns:

- `provider`, `region`, `instance_type`
- `start_time_utc`, `created_at`
- `duration_minutes`, `interrupted`
- `outcome`

Optional:

- `features_snapshot`  
  A JSON string containing launch-time metadata (for example, launch hour or price features).

---

## Running the model

From the **repository root**, run:  
python3 -m src.cai_model.cli
--csv-path data/probe_results_rows.csv
--block-hours 6
--outdir outputs


### Arguments

- `--csv-path`  
  Path to the probe results CSV file.

- `--block-hours`  
  Forward-looking horizon (in hours) over which termination risk is computed.

- `--outdir`  
  Directory where plots and CSV outputs are written.

---

## Outputs

The pipeline produces:

- Learning curve plots:
  - `learning_curve_roc.png`
  - `learning_curve_brier.png`
- `best_instances.csv`  
  A subset of training slice-instances selected via random subset search that yields the best test performance.
- Printed evaluation tables:
  - Reliability table
  - Per-pool Brier score comparisons
  - Per-block actual vs predicted termination rates

