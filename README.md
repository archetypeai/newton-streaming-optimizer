# Newton Machine State Optimizer for Streaming

Grid-search optimizer for [Newton](https://www.archetypeai.dev/) Machine State Lens streaming configurations. Brute-forces KNN hyperparameters via the streaming API and outputs the best config based on F1 score.

**For batch optimization, see [archetype-batch-examples](https://github.com/archetypeai/archetype-batch-examples#7-config-optimization).**

## How It Works

1. **Uploads n-shot files** (one CSV per class) — cached across configs
2. **Finds mixed data sections** in the inference file — ensures balanced ground truth for evaluation
3. **For each config combination:**
   - Creates a Machine State Lens session
   - Sends a probe window and waits for warm-up (~60-90s per session)
   - Streams 40 inference windows at 1s intervals
   - Collects predictions via SSE
   - Compares against ground truth (unanimous windows only)
   - Computes F1 score, precision, recall per class
4. **Ranks all configs by macro F1 score**
5. **Outputs** best config as a ready-to-use lens config JSON

## Parameter Grid

| Parameter | Default Values | Description |
|-----------|---------------|-------------|
| `window_size` | 32, 64, 128 | Samples per classification window |
| `n_neighbors` | 3, 5, 7 | KNN neighbor count |
| `metric` | euclidean, manhattan, cosine | KNN distance metric |
| `weights` | uniform, distance | KNN weighting scheme |

Default grid: 3 × 3 × 3 × 2 = **54 configurations** (~90 min total at ~100s per config).

## Setup

```bash
pip install -r requirements.txt

# Configure API key
export ATAI_API_KEY=your_api_key_here
```

## Usage

### Basic

```bash
python optimize.py \
    --inference-file data/inference.csv \
    --n-shot-files data/class_a.csv data/class_b.csv \
    --class-names CLASS_A CLASS_B \
    --data-columns sensor_1 sensor_2 sensor_3 \
    --timestamp-column timestamp \
    --label-column label
```

### Volve Drilling Example

```bash
python optimize.py \
    --inference-file /path/to/volve_csv/Norway-StatoilHydro-15_9-F-14.csv \
    --n-shot-files /path/to/volve_drilling.csv /path/to/volve_not_drilling.csv \
    --class-names DRILLING NOT_DRILLING \
    --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \
    --timestamp-column DATE_TIME \
    --label-column ACTC_LABEL \
    --window-sizes 64 128 \
    --k-values 3 5 \
    --metrics euclidean manhattan
```

### Custom Grid

```bash
python optimize.py \
    --inference-file data.csv \
    --n-shot-files healthy.csv broken.csv warning.csv \
    --class-names HEALTHY BROKEN WARNING \
    --data-columns accel_x accel_y accel_z temperature \
    --timestamp-column ts \
    --label-column state \
    --window-sizes 64 128 256 512 \
    --k-values 3 5 7 10 15 \
    --metrics euclidean cosine \
    --windows-per-config 60 \
    --probe-timeout 120
```

## Output

### optimizer_results.json

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

### best_config.json

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
| `--windows-per-config` | No | 40 | Inference windows per config |
| `--probe-timeout` | No | 90 | Probe warm-up timeout (seconds) |

## Comparison: Streaming vs Batch Optimization

| | This Tool (Streaming) | [Batch Optimizer](https://github.com/archetypeai/archetype-batch-examples#7-config-optimization) |
|---|---|---|
| **API** | Lens session + SSE | Batch job API |
| **Speed** | ~100s per config (live) | Minutes per config (async) |
| **Model** | `omega_embeddings_01` | `omega_1_3_surface` |
| **Warm-up** | ~60-90s per session | None (batch processed) |
| **Data flow** | Stream windows in real-time | Upload full file, process server-side |
| **Use case** | Optimize for real-time streaming apps | Optimize for batch processing pipelines |

## Known Limitations

- **Generic encoder**: `omega_embeddings_01` may not produce discriminative embeddings for all domains. Domain-specific encoders (like `omega_1_3_surface` for drilling) significantly improve accuracy but are currently only available for batch processing.
- **Session cold start**: Each config requires ~60-90s for Newton to process n-shot examples. For 54 configs, total runtime is ~90 minutes.
- **SSE reliability**: Long-running optimizations may experience SSE disconnections. The script handles reconnection but some windows may be lost.
