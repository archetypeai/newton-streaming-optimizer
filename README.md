# Newton Machine State Optimizer for Streaming

Grid-search optimizer for [Newton](https://www.archetypeai.dev/) Machine State Lens streaming configurations. Brute-forces KNN hyperparameters via the streaming API and outputs the best config based on F1 score.

**For batch optimization, see [archetype-batch-examples](https://github.com/archetypeai/archetype-batch-examples#7-config-optimization).**

## Workflow

```
labeled CSV  ──►  prep_data.py  ──►  inference.csv + nshot_<class>.csv  ──►  optimize.py  ──►  best_config.json
```

1. **Bring a labeled time-series CSV** — one row per sample, one column with the ground-truth class label.
2. **Run `prep_data.py`** to split it into n-shot files (one per class) and a balanced inference file.
3. **Run `optimize.py`** to grid-search KNN configs and pick the best one by F1.

> **No labeled data?** Skip step 2 — pre-prepared drilling examples live in [`examples/drilling/`](examples/drilling) (derived from the [Volve dataset](https://www.equinor.com/energy/volve-data-sharing)). Jump straight to [Quick Start](#quick-start).

## Setup

```bash
pip install -r requirements.txt
export ATAI_API_KEY=your_api_key_here
```

## Quick Start

Run the optimizer against the bundled drilling example — no data prep required:

```bash
python optimize.py \
    --inference-file examples/drilling/inference.csv \
    --n-shot-files examples/drilling/nshot_drilling.csv examples/drilling/nshot_not_drilling.csv \
    --class-names drilling not_drilling \
    --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \
    --timestamp-column DATE_TIME \
    --label-column label
```

This searches the default 54-config grid (~90 min) and writes `optimizer_results.json` + `best_config.json`.

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
   - Sends a probe window and waits for warm-up (~60-90s)
   - Streams 100 inference windows at 1s intervals, collecting predictions via SSE
   - Compares against ground truth (unanimous windows only)
   - Computes F1, precision, recall per class
4. Ranks all configs by macro F1 score
5. Writes the best config as a ready-to-use lens config JSON

## Parameter Grid

| Parameter | Default Values | Description |
|-----------|---------------|-------------|
| `window_size` | 32, 64, 128 | Samples per classification window |
| `n_neighbors` | 3, 5, 7 | KNN neighbor count |
| `metric` | euclidean, manhattan, cosine | KNN distance metric |
| `weights` | uniform, distance | KNN weighting scheme |

Default grid: 3 × 3 × 3 × 2 = **54 configurations** (~90 min total at ~100s per config).

## Output

### `optimizer_results.json`

Full results with all configs ranked by F1:

```json
{
  "best_config": {
    "window_size": 128,
    "n_neighbors": 5,
    "metric": "euclidean",
    "weights": "uniform"
  },
  "best_metrics": {
    "macro_f1": 0.67,
    "accuracy": 0.72,
    "per_class": {
      "DRILLING": { "precision": 0.8, "recall": 0.6, "f1": 0.69 },
      "NOT_DRILLING": { "precision": 0.65, "recall": 0.83, "f1": 0.73 }
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
    "timestamp_column": "timestamp",
    "data_columns": ["sensor_1", "sensor_2", "sensor_3"],
    "window_size": 128,
    "step_size": 128
  },
  "knn_configs": {
    "n_neighbors": 5,
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

`examples/drilling/` was generated from the [Volve dataset](https://www.equinor.com/energy/volve-data-sharing) (7.3M labeled rows) using:

```bash
python prep_data.py \
    --input-file volve_raw_labeled.csv \
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

- **Generic encoder**: `omega_embeddings_01` may not produce discriminative embeddings for all domains. Domain-specific encoders (like `omega_1_3_surface` for drilling) significantly improve accuracy but are currently only available for batch processing.
- **Session cold start**: Each config requires ~60-90s for Newton to process n-shot examples. For 54 configs, total runtime is ~3 hours.
- **SSE reliability**: Long-running optimizations may experience SSE disconnections. The script handles reconnection but some windows may be lost.
