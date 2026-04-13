# Newton Machine State Optimizer for Streaming

Grid-search optimizer for [Newton](https://www.archetypeai.dev/) Machine State Lens streaming configurations. Brute-forces KNN hyperparameters via the streaming API and outputs the best config based on F1 score.

**For batch optimization, see [archetype-batch-examples](https://github.com/archetypeai/archetype-batch-examples#7-config-optimization).**

## Workflow

```
labeled CSV  ──►  prep_data.py  ──►  inference.csv + nshot_<class>.csv  ──►  optimize.py  ──►  best_config.json  ──►  classify.py  ──►  predictions.csv
```

1. **Bring a labeled time-series CSV** — one row per sample, one column with the ground-truth class label.
2. **Run `prep_data.py`** to split it into n-shot files (one per class) and a balanced inference file.
3. **Run `optimize.py`** to grid-search KNN configs and pick the best one by F1.
4. **Run `classify.py`** with the winning `best_config.json` to classify any new (unlabeled or labeled) CSV.

> **No labeled data?** Skip step 2 — pre-prepared drilling examples live in [`examples/drilling/`](examples/drilling) (derived from the [Volve dataset](https://www.equinor.com/energy/volve-data-sharing)). Jump straight to [Quick Start](#quick-start).

## Setup

```bash
# Create a virtual environment
python3 -m venv myenv

# Activate it
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
export ATAI_API_KEY=your_api_key_here

# Deactivate when done
deactivate
```

## Quick Start

No data prep required — both commands below use the bundled drilling example in [`examples/drilling/`](examples/drilling).

### Find the best config

```bash
python optimize.py \
    --inference-file examples/drilling/inference.csv \
    --n-shot-files examples/drilling/nshot_drilling.csv examples/drilling/nshot_not_drilling.csv \
    --class-names drilling not_drilling \
    --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \
    --timestamp-column DATE_TIME \
    --label-column label
```

Searches the default 54-config grid and writes `optimizer_results.json` + `best_config.json`. Runtime varies — small grids and dead-metric / weights no-op auto-skipping typically bring it to ~30–40 minutes.

### Classify with the winning config

Once `best_config.json` exists, run `classify.py` against the same example data to see it in action end-to-end:

```bash
python classify.py \
    --config-file best_config.json \
    --n-shot-files examples/drilling/nshot_drilling.csv examples/drilling/nshot_not_drilling.csv \
    --class-names drilling not_drilling \
    --inference-file examples/drilling/inference.csv \
    --label-column label \
    --output predictions.csv
```

Writes `predictions.csv` and prints macro F1 + per-class precision/recall at the end (the bundled inference file is labeled, so evaluation kicks in automatically). Real output from this exact command:

```
Evaluation (1,465 unanimous-window pairs):
  macro F1: 94.6%
  accuracy: 94.6%
  drilling:     P=91.1% R=98.1% F1=94.4% (TP=672 FP=66 FN=13)
  not_drilling: P=98.2% R=91.5% F1=94.8% (TP=714 FP=13 FN=66)
```

## Use Your Own Data

### Step 1 — Prep

```bash
python prep_data.py \
    --input-file your_labeled.csv \
    --output-dir data/prepared \
    --data-columns sensor_1 sensor_2 sensor_3 \
    --timestamp-column timestamp \
    --label-column label
```

`prep_data.py` picks the **longest contiguous run** of each class for the n-shot files and the **most class-balanced contiguous slice** for the inference file. It prints the exact `optimize.py` command to run next.

If your labels are raw codes that need remapping, use `--label-mapping`:

```bash
python prep_data.py \
    --input-file raw.csv \
    --output-dir data/prepared \
    --data-columns sensor_1 sensor_2 sensor_3 \
    --label-column ACTC \
    --label-mapping "1:DRILLING,2:DRILLING,3:NOT_DRILLING,4:NOT_DRILLING"
```

### Step 2 — Optimize

Use the command `prep_data.py` printed, or customize the grid:

```bash
python optimize.py \
    --inference-file data/prepared/inference.csv \
    --n-shot-files data/prepared/nshot_healthy.csv data/prepared/nshot_broken.csv \
    --class-names HEALTHY BROKEN \
    --data-columns accel_x accel_y accel_z temperature \
    --timestamp-column ts \
    --label-column label \
    --window-sizes 64 128 256 \
    --k-values 3 5 7 \
    --metrics euclidean cosine \
    --windows-per-config 60 \
    --probe-timeout 120
```

### Step 3 — Classify new data with the winning config

Once `optimize.py` writes `best_config.json`, use `classify.py` to apply it to any CSV:

```bash
python classify.py \
    --config-file best_config.json \
    --n-shot-files data/prepared/nshot_drilling.csv data/prepared/nshot_not_drilling.csv \
    --class-names drilling not_drilling \
    --inference-file data/full_well.csv \
    --output predictions.csv
```

If your inference CSV has ground-truth labels, add `--label-column` and `classify.py` will print evaluation metrics (F1, precision, recall, confusion matrix) at the end:

```bash
python classify.py \
    --config-file best_config.json \
    --n-shot-files data/prepared/nshot_drilling.csv data/prepared/nshot_not_drilling.csv \
    --class-names drilling not_drilling \
    --inference-file data/labeled_well.csv \
    --label-column label \
    --output predictions.csv
```

Output (`predictions.csv`):

```csv
window_index,DATE_TIME,prediction,ground_truth
0,1182186370,not_drilling,not_drilling
1,1182186402,not_drilling,not_drilling
2,1182186434,drilling,drilling
...
```

Console summary (when `--label-column` is supplied) — example from running against the bundled `examples/drilling/inference.csv` (1,562 windows × 128 samples):

```
Evaluation (1,465 unanimous-window pairs):
  macro F1: 94.6%
  accuracy: 94.6%
  drilling:     P=91.1% R=98.1% F1=94.4% (TP=672 FP=66 FN=13)
  not_drilling: P=98.2% R=91.5% F1=94.8% (TP=714 FP=13 FN=66)
```

> Note: this is a more realistic accuracy estimate than the optimizer's headline 100% — the optimizer evaluates 100 windows against the most-balanced section it can find, while `classify.py` here scores 1,465 unanimous windows across the full 200K-row inference file. F1 ≈ 95% over a much broader and more varied evaluation is the number to plan around.

## How It Works

**`prep_data.py`** does a single streaming pass over your labeled CSV:

1. Scans the label column, applying optional `--label-mapping`
2. Finds the longest contiguous run of each class → writes `nshot_<class>.csv` (sensor columns only)
3. Sliding-window search for the most class-balanced contiguous slice → writes `inference.csv` (sensor columns + label)

**`optimize.py`** searches the parameter grid:

1. Uploads n-shot files once — cached across all configs
2. Finds mixed sections in the inference file for balanced ground-truth evaluation
3. For each config combination:
   - Creates a Machine State Lens session
   - Sends a probe window and sleeps 60s so Newton's KNN index finishes loading
   - Streams windows, collects predictions via SSE
   - Compares against ground truth (unanimous windows only) for F1
4. Auto-detects and skips dead configs to save time:
   - **Dead metrics** (3 consecutive empty runs with no prior success) → all remaining configs at that metric short-circuit
   - **Weights no-op** (3 matching uniform/distance pairs at a metric) → future distance configs at that metric reuse the uniform result
5. Writes `optimizer_results.json` (full leaderboard) and `best_config.json` (top result, lens-API-ready) — but only if at least one config returned predictions, so an API outage won't clobber prior good outputs

**`classify.py`** runs a single Machine State Lens session using a saved config:

1. Loads `best_config.json` (or any compatible lens config JSON)
2. Uploads n-shot files
3. Streams the inference CSV through Newton in `window_size`-sized windows
4. Writes `predictions.csv` with `window_index`, optional timestamp, and predicted class
5. If `--label-column` is supplied, also prints macro F1 + per-class precision/recall

## Parameter Grid

| Parameter | Default Values | Description |
|-----------|---------------|-------------|
| `window_size` | 32, 64, 128 | Samples per classification window |
| `n_neighbors` | 3, 5, 7 | KNN neighbor count |
| `metric` | euclidean, manhattan, cosine | KNN distance metric |
| `weights` | uniform, distance | KNN weighting scheme |

Default grid: 3 × 3 × 3 × 2 = **54 configurations**. Total runtime depends heavily on which metrics/weights survive auto-skipping. On the bundled drilling example: cosine returns no predictions on `omega_embeddings_01` (all 18 configs auto-skipped after detection), and `weights=distance` is a no-op for euclidean (6 more configs reuse uniform results) — so the effective grid is ~30 active configs at ~120s each ≈ **~60 min**.

## Output

### `optimizer_results.json`

Full results with all configs ranked by F1. Example (real numbers from running on `examples/drilling/`):

```json
{
  "best_config": {
    "window_size": 128,
    "n_neighbors": 3,
    "metric": "euclidean",
    "weights": "uniform"
  },
  "best_metrics": {
    "macro_f1": 1.0,
    "accuracy": 1.0,
    "per_class": {
      "drilling":     { "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 79, "fp": 0, "fn": 0 },
      "not_drilling": { "precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 20, "fp": 0, "fn": 0 }
    }
  },
  "all_results": [ ... ]
}
```

### `best_config.json`

Ready-to-use lens config for the Newton streaming API:

```json
{
  "model_name": "OmegaEncoder",
  "model_version": "OmegaEncoder::omega_embeddings_01",
  "normalize_input": true,
  "buffer_size": 128,
  "csv_configs": {
    "timestamp_column": "DATE_TIME",
    "data_columns": ["BPOS", "DBTM", "FLWI", "HDTH", "HKLD", "ROP", "RPM", "SPPA", "WOB"],
    "window_size": 128,
    "step_size": 128
  },
  "knn_configs": {
    "n_neighbors": 3,
    "metric": "euclidean",
    "weights": "uniform",
    "algorithm": "ball_tree",
    "normalize_embeddings": false
  }
}
```

## CLI Reference

### `prep_data.py`

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input-file` | Yes | — | Source labeled CSV |
| `--data-columns` | Yes | — | Sensor column names |
| `--label-column` | Yes | — | Ground-truth label column |
| `--output-dir` | No | `prepared_data` | Where to write outputs |
| `--timestamp-column` | No | — | Optional timestamp column to carry through |
| `--label-mapping` | No | — | Remap raw labels, e.g. `"1:DRILLING,2:DRILLING,3:NOT_DRILLING"` |
| `--classes` | No | all found | Limit to these class names (after mapping) |
| `--n-shot-size` | No | 2000 | Rows per n-shot file |
| `--inference-size` | No | 200000 | Rows in the inference slice (headroom for window sizes up to 1024) |

### `classify.py`

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--config-file` | Yes | — | Lens config JSON (`best_config.json` from `optimize.py`) |
| `--n-shot-files` | Yes | — | One CSV per class |
| `--class-names` | Yes | — | Class names matching n-shot files |
| `--inference-file` | Yes | — | CSV to classify |
| `--data-columns` | No | from config | Override data columns |
| `--timestamp-column` | No | from config | Override timestamp column |
| `--label-column` | No | — | Optional ground-truth column for evaluation |
| `--output` | No | `predictions.csv` | Output CSV path |
| `--api-key` | No | `$ATAI_API_KEY` | API key |
| `--api-endpoint` | No | `https://api.u1.archetypeai.app` | API endpoint |
| `--probe-timeout` | No | 90 | Probe warm-up timeout (seconds) |
| `--stream-delay` | No | 0.5 | Delay between window sends (seconds) |
| `--max-windows` | No | whole file | Stop after this many windows |

### `optimize.py`

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--inference-file` | Yes | — | CSV file with sensor data + ground truth labels |
| `--n-shot-files` | Yes | — | One CSV per class (no label column, sensor data only) |
| `--class-names` | Yes | — | Class names matching each n-shot file |
| `--data-columns` | Yes | — | Sensor column names to use |
| `--timestamp-column` | No | `timestamp` | Timestamp column name |
| `--label-column` | Yes | — | Ground truth label column in inference file |
| `--window-sizes` | No | 32 64 128 | Window sizes to search |
| `--k-values` | No | 3 5 7 | K neighbor values to search |
| `--metrics` | No | euclidean manhattan cosine | Distance metrics to search |
| `--api-key` | No | `$ATAI_API_KEY` | API key |
| `--api-endpoint` | No | `https://api.u1.archetypeai.app` | API endpoint |
| `--output` | No | `optimizer_results.json` | Full results output file |
| `--config-output` | No | `best_config.json` | Best config output file |
| `--windows-per-config` | No | 100 | Inference windows per config |
| `--probe-timeout` | No | 90 | Probe warm-up timeout (seconds) |

## Bundled Example: Drilling

`examples/drilling/` was generated from the [Volve dataset](https://www.equinor.com/energy/volve-data-sharing) (7.3M labeled rows). The source `volve_raw_labeled.csv` ships via Git LFS with [archetype-batch-examples](https://github.com/archetypeai/archetype-batch-examples) — clone it alongside this repo and run:

```bash
python prep_data.py \
    --input-file ../archetype-batch-examples/data/volve_raw_labeled.csv \
    --output-dir examples/drilling \
    --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \
    --timestamp-column DATE_TIME \
    --label-column label \
    --classes drilling not_drilling
```

| File | Rows | Contents |
|------|------|----------|
| `inference.csv` | 200,000 | 50% drilling, 50% not_drilling — balanced ground truth |
| `nshot_drilling.csv` | 2,000 | Sensor-only n-shot examples for the `drilling` class |
| `nshot_not_drilling.csv` | 2,000 | Sensor-only n-shot examples for the `not_drilling` class |

### Reference results

Top of the leaderboard from running `optimize.py` against this bundled example (full grid, ~60 min):

| Rank | Config | F1 | Acc |
|---:|---|---:|---:|
| 1 | **w128 k3 euc unif** | **100.0%** | **100.0%** |
| 3 | w64 k3/k5 euc | 97.1% | 97.9-98.0% |
| 7 | w128 k5 euc | 97.0% | 98.0% |
| 19 | w64 k3-k7 manhattan | 93.0% | 94.9% |
| 25 | w32 k3 euc | 85.4% | 88.0% |
| 37+ | all cosine | 0% | 0% (auto-skipped) |

Patterns: **euclidean > manhattan** consistently (+2-4 F1), **larger window monotonically better** (w32 → w64 → w128 = 85 → 97 → 100), **k=3 best at w128**, **`weights` is effectively a no-op for euclidean**. **Cosine returns no predictions on `omega_embeddings_01`** — likely a platform-level constraint.

> The 100% headline is in-distribution evaluation on the most-balanced section the optimizer could find, scored only on unanimous windows. Real-world per-well numbers will be lower.

## Data Attribution

The drilling sensor data used in these examples is from the **Equinor Volve Data Village**, released under a modified CC BY 4.0 license. The data may be used for commercial and non-commercial purposes but may not be resold.

> Data provided by Equinor and the former Volve license partners (ExxonMobil Exploration & Production Norway AS and Bayerngas Norge AS). [Terms and Conditions](https://www.equinor.com/energy/volve-data-sharing).

## Comparison: Streaming vs Batch Optimization

| | This Tool (Streaming) | [Batch Optimizer](https://github.com/archetypeai/archetype-batch-examples#7-config-optimization) |
|---|---|---|
| **API** | Lens session + SSE | Batch job API |
| **Speed** | ~3 min per config (live) | Minutes per config (async) |
| **Model** | `omega_embeddings_01` | `omega_1_3_surface` |
| **Warm-up** | ~60-90s per session | None (batch processed) |
| **Data flow** | Stream windows in real-time | Upload full file, process server-side |
| **Use case** | Optimize for real-time streaming apps | Optimize for batch processing pipelines |

## Known Limitations

- **Generic encoder**: `omega_embeddings_01` may not produce discriminative embeddings for all domains. Domain-specific encoders (like `omega_1_3_surface` for drilling) significantly improve accuracy on the broader distribution but are currently only available for batch processing.
- **Cosine metric returns no predictions on `omega_embeddings_01`**. The optimizer auto-detects this (3 consecutive empty runs at a metric → flagged dead, remaining configs skipped), but be aware that any cosine row in the leaderboard will show F1=0% / 0 windows.
- **Session settle time**: Each config sleeps 60s after sending a probe so Newton's KNN index finishes building from the n-shot files. Skipping this drops F1 dramatically (~85% → ~50% on the bundled drilling slice). The 60s figure is the empirically-determined floor — going lower hits a half-built index.
- **In-distribution evaluation**: The optimizer evaluates against a slice of the same labeled CSV used to derive n-shot examples. Real-world performance on different operators / equipment / geologies will be lower than the leaderboard numbers.
- **SSE reliability**: Long-running optimizations may experience SSE disconnections. The persistent `SSEReader` handles this with a single connection per session, but transient API outages will still abort individual configs.
