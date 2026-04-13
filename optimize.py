#!/usr/bin/env python3
"""
Newton Machine State Optimizer for Streaming

Brute-forces a grid of Machine State Lens parameters via the streaming API
to find the best configuration for your time-series classification task.

Usage:
    python optimize.py \
        --inference-file data/inference.csv \
        --n-shot-files data/class_a.csv data/class_b.csv \
        --class-names CLASS_A CLASS_B \
        --data-columns col1 col2 col3 \
        --timestamp-column timestamp \
        --label-column label

Requires: ATAI_API_KEY environment variable (or --api-key flag)
"""

import argparse
import csv
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import requests
from sseclient import SSEClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Configuration ---

API_VERSION = "v0.5"
DEFAULT_ENDPOINT = "https://api.u1.archetypeai.app"

GRID = {
    "window_size": [32, 64, 128],
    "n_neighbors": [3, 5, 7],
    "metric": ["euclidean", "manhattan", "cosine"],
    "weights": ["uniform", "distance"],
}

WINDOWS_PER_CONFIG = 100
PROBE_TIMEOUT_SEC = 90
STREAM_DELAY_SEC = 1.0


# --- API Helpers ---

def api_url(endpoint, path):
    return f"{endpoint.rstrip('/')}/{API_VERSION}{path}"


def api_get(endpoint, api_key, path):
    r = requests.get(api_url(endpoint, path), headers={"Authorization": f"Bearer {api_key}"})
    r.raise_for_status()
    return r.json()


def api_post(endpoint, api_key, path, body, timeout=30):
    r = requests.post(
        api_url(endpoint, path),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def upload_file(endpoint, api_key, file_path):
    with open(file_path, "rb") as f:
        r = requests.post(
            api_url(endpoint, "/files"),
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (Path(file_path).name, f, "text/csv")},
        )
    r.raise_for_status()
    return r.json()["file_id"]


# --- Data Helpers ---

def load_csv(file_path, data_columns, label_column=None):
    """Load CSV and return rows as list of dicts."""
    rows = []
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def find_mixed_offset(rows, label_column, class_names, window_size, num_windows=41):
    """Find the section with the best mix of classes."""
    scan_size = num_windows * window_size
    best_offset = 0
    best_mix = 0.0
    step = max(1, window_size * 10)

    for offset in range(0, len(rows) - scan_size + 1, step):
        counts = {name: 0 for name in class_names}
        for i in range(offset, offset + scan_size):
            label = rows[i].get(label_column, "").strip()
            if label in counts:
                counts[label] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        minority = min(counts.values())
        mix = minority / total
        if mix > best_mix:
            best_mix = mix
            best_offset = offset
        if best_mix >= 0.45:
            break

    return best_offset, best_mix


def get_window_ground_truth(rows, start, window_size, label_column, class_names):
    """Get unanimous ground truth for a window. Returns class name or None if mixed."""
    counts = {name: 0 for name in class_names}
    for i in range(start, min(start + window_size, len(rows))):
        label = rows[i].get(label_column, "").strip()
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    for name, count in counts.items():
        if count == total:
            return name
    return None  # mixed


def transpose_window(rows, start, window_size, data_columns):
    """Transpose rows to channel-first format for Newton API."""
    channels = []
    for col in data_columns:
        values = []
        for i in range(start, min(start + window_size, len(rows))):
            val = rows[i].get(col, "")
            try:
                values.append(float(val))
            except (ValueError, TypeError):
                values.append(0.0)
        channels.append(values)
    return channels


# --- Lens Session Management ---

def clean_stale_lenses(endpoint, api_key, lens_prefix="optimizer-lens"):
    lenses = api_get(endpoint, api_key, "/lens/metadata")
    if isinstance(lenses, list):
        for lens in lenses:
            if lens.get("lens_name", "").startswith(lens_prefix):
                try:
                    api_post(endpoint, api_key, "/lens/delete", {"lens_id": lens["lens_id"]})
                except Exception:
                    pass


def create_session(endpoint, api_key, n_shot_map, config, data_columns, timestamp_column):
    """Create a Machine State Lens session with the given config."""
    lens_name = f"optimizer-lens-{int(time.time())}"

    lens_config = {
        "lens_name": lens_name,
        "lens_config": {
            "model_pipeline": [
                {"processor_name": "lens_timeseries_state_processor", "processor_config": {}}
            ],
            "model_parameters": {
                "model_name": "OmegaEncoder",
                "model_version": "OmegaEncoder::omega_embeddings_01",
                "normalize_input": True,
                "buffer_size": config["window_size"],
                "input_n_shot": n_shot_map,
                "csv_configs": {
                    "timestamp_column": timestamp_column,
                    "data_columns": data_columns,
                    "window_size": config["window_size"],
                    "step_size": config["window_size"],
                },
                "knn_configs": {
                    "n_neighbors": config["n_neighbors"],
                    "metric": config["metric"],
                    "weights": config["weights"],
                    "algorithm": "ball_tree",
                    "normalize_embeddings": False,
                },
            },
            "output_streams": [{"stream_type": "server_sent_events_writer"}],
        },
    }

    lens = api_post(endpoint, api_key, "/lens/register", {"lens_config": lens_config}, timeout=60)
    session = api_post(endpoint, api_key, "/lens/sessions/create", {"lens_id": lens["lens_id"]})
    session_id = session["session_id"]

    # Wait for session ready
    start_time = time.time()
    while time.time() - start_time < 60:
        status = api_post(
            endpoint, api_key, "/lens/sessions/events/process",
            {"session_id": session_id, "event": {"type": "session.status"}},
            timeout=10,
        )
        s = status.get("session_status", "")
        if s in ("LensSessionStatus.SESSION_STATUS_RUNNING", "3"):
            break
        if s in ("LensSessionStatus.SESSION_STATUS_FAILED", "6"):
            raise RuntimeError(f"Session failed: {s}")
        time.sleep(1)

    sse_url = api_url(endpoint, f"/lens/sessions/consumer/{session_id}")
    return session_id, sse_url


def stream_window(endpoint, api_key, session_id, sensor_data, counter):
    """Send a data window to the session."""
    api_post(
        endpoint, api_key, "/lens/sessions/events/process",
        {
            "session_id": session_id,
            "event": {
                "type": "session.update",
                "event_data": {
                    "type": "data.json",
                    "event_data": {
                        "sensor_data": sensor_data,
                        "sensor_metadata": {
                            "sensor_timestamp": time.time(),
                            "sensor_id": f"optimizer_{counter}",
                        },
                    },
                },
            },
        },
        timeout=15,
    )


def destroy_session(endpoint, api_key, session_id):
    try:
        api_post(endpoint, api_key, "/lens/sessions/destroy", {"session_id": session_id})
    except Exception:
        pass


# --- SSE Consumer ---

def collect_sse_results(sse_url, api_key, timeout_sec=10):
    """Collect SSE results for a short period."""
    results = []
    try:
        r = requests.get(sse_url, headers={"Authorization": f"Bearer {api_key}"}, stream=True, timeout=timeout_sec)
        client = SSEClient(r)
        start = time.time()
        for event in client.events():
            if time.time() - start > timeout_sec:
                break
            try:
                data = json.loads(event.data)
                if data.get("type") == "inference.result":
                    response = data.get("event_data", {}).get("response")
                    if isinstance(response, str):
                        results.append(response)
                    elif isinstance(response, list):
                        results.append(response[0])
                    elif isinstance(response, dict):
                        results.append(
                            response.get("class_name")
                            or response.get("label")
                            or response.get("prediction")
                            or str(response)
                        )
            except (json.JSONDecodeError, KeyError):
                pass
    except Exception as e:
        print(f"  SSE error: {e}", file=sys.stderr)
    return results


# --- Evaluation ---

def compute_f1(predictions, ground_truths, class_names):
    """Compute F1 score for multi-class (per-class, then macro average)."""
    per_class = {}
    for cls in class_names:
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g == cls)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p == cls and g != cls)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if p != cls and g == cls)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(class_names) if class_names else 0
    accuracy = sum(1 for p, g in zip(predictions, ground_truths) if p == g) / len(predictions) if predictions else 0

    return {"macro_f1": macro_f1, "accuracy": accuracy, "per_class": per_class}


# --- Main Optimizer ---

def run_optimizer(args):
    endpoint = args.api_endpoint
    api_key = args.api_key or os.getenv("ATAI_API_KEY")
    if not api_key:
        print("Error: ATAI_API_KEY not set. Use --api-key or set the environment variable.", file=sys.stderr)
        sys.exit(1)

    # Load data
    print(f"Loading inference data: {args.inference_file}")
    rows = load_csv(args.inference_file, args.data_columns, args.label_column)
    print(f"  {len(rows)} rows, columns: {args.data_columns}")

    # Upload n-shot files
    print("Uploading n-shot files...")
    n_shot_map = {}
    for file_path, class_name in zip(args.n_shot_files, args.class_names):
        print(f"  Uploading {file_path} as {class_name}...")
        file_id = upload_file(endpoint, api_key, file_path)
        n_shot_map[class_name] = file_id
        print(f"  → {file_id}")

    # Find mixed data section
    print("Finding mixed data section...")
    class_names = args.class_names

    # Build grid
    grid_keys = list(GRID.keys())
    grid_values = list(GRID.values())

    # Override grid if user specified
    if args.window_sizes:
        GRID["window_size"] = args.window_sizes
        grid_values[grid_keys.index("window_size")] = args.window_sizes
    if args.k_values:
        GRID["n_neighbors"] = args.k_values
        grid_values[grid_keys.index("n_neighbors")] = args.k_values
    if args.metrics:
        GRID["metric"] = args.metrics
        grid_values[grid_keys.index("metric")] = args.metrics

    configs = [dict(zip(grid_keys, combo)) for combo in product(*grid_values)]
    total = len(configs)
    print(f"\nGrid search: {total} configurations")
    print(f"  window_size: {GRID['window_size']}")
    print(f"  n_neighbors: {GRID['n_neighbors']}")
    print(f"  metric: {GRID['metric']}")
    print(f"  weights: {GRID['weights']}")
    print(f"  windows per config: {WINDOWS_PER_CONFIG}")
    print()

    # Clean stale lenses
    clean_stale_lenses(endpoint, api_key)

    results = []

    for idx, config in enumerate(configs):
        label = f"w{config['window_size']} k{config['n_neighbors']} {config['metric'][:3]} {config['weights'][:4]}"
        print(f"[{idx+1}/{total}] {label}")

        # Find mixed offset for this window size
        offset, mix = find_mixed_offset(rows, args.label_column, class_names, config["window_size"])
        print(f"  Data offset: {offset} (mix: {mix*100:.1f}%)")

        try:
            # Create session
            print("  Creating session...")
            session_id, sse_url = create_session(
                endpoint, api_key, n_shot_map, config,
                args.data_columns, args.timestamp_column,
            )

            # Send probe and wait for warm-up
            print("  Warming up (probe)...")
            probe_data = transpose_window(rows, offset, config["window_size"], args.data_columns)
            stream_window(endpoint, api_key, session_id, probe_data, 0)

            # Start SSE consumer in background and wait for first result
            warmed_up = False
            probe_start = time.time()
            while time.time() - probe_start < PROBE_TIMEOUT_SEC:
                probe_results = collect_sse_results(sse_url, api_key, timeout_sec=5)
                if probe_results:
                    warmed_up = True
                    print(f"  Warm-up complete: {probe_results[0]}")
                    break

            if not warmed_up:
                print(f"  Warm-up timed out after {PROBE_TIMEOUT_SEC}s")

            # Stream inference windows and collect results
            print(f"  Streaming {WINDOWS_PER_CONFIG} windows...")
            predictions = []
            ground_truths = []

            for i in range(1, WINDOWS_PER_CONFIG + 1):
                start = offset + i * config["window_size"]
                if start + config["window_size"] > len(rows):
                    break

                # Get ground truth
                gt = get_window_ground_truth(rows, start, config["window_size"], args.label_column, class_names)

                # Stream window
                sensor_data = transpose_window(rows, start, config["window_size"], args.data_columns)
                stream_window(endpoint, api_key, session_id, sensor_data, i)
                time.sleep(STREAM_DELAY_SEC)

                # Collect results periodically
                if i % 5 == 0 or i == WINDOWS_PER_CONFIG:
                    batch_results = collect_sse_results(sse_url, api_key, timeout_sec=3)
                    for result in batch_results:
                        if gt is not None:
                            predictions.append(result)
                            ground_truths.append(gt)

            # Compute metrics
            if predictions:
                metrics = compute_f1(predictions, ground_truths, class_names)
                print(f"  Results: F1={metrics['macro_f1']*100:.1f}% Acc={metrics['accuracy']*100:.1f}% ({len(predictions)} scored windows)")
                for cls, stats in metrics["per_class"].items():
                    print(f"    {cls}: P={stats['precision']*100:.1f}% R={stats['recall']*100:.1f}% F1={stats['f1']*100:.1f}% (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})")
            else:
                metrics = {"macro_f1": 0, "accuracy": 0, "per_class": {}}
                print("  No results received")

            results.append({
                "config": config,
                "label": label,
                "metrics": metrics,
                "scored_windows": len(predictions),
                "offset": offset,
                "mix": mix,
            })

            # Cleanup
            destroy_session(endpoint, api_key, session_id)

        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            results.append({
                "config": config,
                "label": label,
                "metrics": {"macro_f1": 0, "accuracy": 0, "per_class": {}},
                "scored_windows": 0,
                "error": str(e),
            })

        print()

    # Rank by F1
    results.sort(key=lambda r: r["metrics"]["macro_f1"], reverse=True)

    # Print leaderboard
    print("=" * 70)
    print("LEADERBOARD (sorted by F1)")
    print("=" * 70)
    for i, r in enumerate(results):
        f1 = r["metrics"]["macro_f1"] * 100
        acc = r["metrics"]["accuracy"] * 100
        print(f"  {i+1:2d}. F1: {f1:5.1f}%  Acc: {acc:5.1f}%  {r['label']:30s}  ({r['scored_windows']}w)")
    print()

    # Best config
    best = results[0]
    print("BEST CONFIG:")
    print(json.dumps(best["config"], indent=2))
    print(f"F1: {best['metrics']['macro_f1']*100:.1f}%  Accuracy: {best['metrics']['accuracy']*100:.1f}%")
    print()

    # Save results
    output_file = args.output or "optimizer_results.json"
    output = {
        "best_config": best["config"],
        "best_metrics": best["metrics"],
        "all_results": results,
        "grid": GRID,
        "settings": {
            "windows_per_config": WINDOWS_PER_CONFIG,
            "probe_timeout_sec": PROBE_TIMEOUT_SEC,
            "stream_delay_sec": STREAM_DELAY_SEC,
            "inference_file": args.inference_file,
            "n_shot_files": dict(zip(args.class_names, args.n_shot_files)),
            "data_columns": args.data_columns,
            "timestamp_column": args.timestamp_column,
            "label_column": args.label_column,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_file}")

    # Save best config separately
    config_file = args.config_output or "best_config.json"
    best_lens_config = {
        "model_name": "OmegaEncoder",
        "model_version": "OmegaEncoder::omega_embeddings_01",
        "normalize_input": True,
        "buffer_size": best["config"]["window_size"],
        "csv_configs": {
            "timestamp_column": args.timestamp_column,
            "data_columns": args.data_columns,
            "window_size": best["config"]["window_size"],
            "step_size": best["config"]["window_size"],
        },
        "knn_configs": {
            "n_neighbors": best["config"]["n_neighbors"],
            "metric": best["config"]["metric"],
            "weights": best["config"]["weights"],
            "algorithm": "ball_tree",
            "normalize_embeddings": False,
        },
    }
    with open(config_file, "w") as f:
        json.dump(best_lens_config, f, indent=2)
    print(f"Best lens config saved to {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Newton Machine State Optimizer for Streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Drilling data optimization
  python optimize.py \\
      --inference-file data/volve_csv/Norway-Statoil-15_9-F-12.csv \\
      --n-shot-files data/volve_drilling.csv data/volve_not_drilling.csv \\
      --class-names DRILLING NOT_DRILLING \\
      --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \\
      --timestamp-column DATE_TIME \\
      --label-column ACTC_LABEL

  # Custom grid
  python optimize.py \\
      --inference-file my_data.csv \\
      --n-shot-files healthy.csv broken.csv \\
      --class-names HEALTHY BROKEN \\
      --data-columns sensor_1 sensor_2 sensor_3 \\
      --timestamp-column timestamp \\
      --label-column state \\
      --window-sizes 64 128 256 \\
      --k-values 3 5 10 \\
      --metrics euclidean cosine
        """,
    )

    global WINDOWS_PER_CONFIG, PROBE_TIMEOUT_SEC

    # Required arguments
    parser.add_argument("--inference-file", required=True, help="Path to inference CSV file")
    parser.add_argument("--n-shot-files", nargs="+", required=True, help="Paths to n-shot CSV files (one per class)")
    parser.add_argument("--class-names", nargs="+", required=True, help="Class names matching n-shot files")
    parser.add_argument("--data-columns", nargs="+", required=True, help="Sensor/data column names")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name (default: timestamp)")
    parser.add_argument("--label-column", required=True, help="Ground truth label column name in inference file")

    # Optional grid overrides
    parser.add_argument("--window-sizes", nargs="+", type=int, help="Window sizes to test (default: 32 64 128)")
    parser.add_argument("--k-values", nargs="+", type=int, help="K neighbor values to test (default: 3 5 7)")
    parser.add_argument("--metrics", nargs="+", help="Distance metrics to test (default: euclidean manhattan cosine)")

    # API config
    parser.add_argument("--api-key", help="Archetype AI API key (or set ATAI_API_KEY env var)")
    parser.add_argument("--api-endpoint", default=DEFAULT_ENDPOINT, help=f"API endpoint (default: {DEFAULT_ENDPOINT})")

    # Output
    parser.add_argument("--output", default="optimizer_results.json", help="Output file for full results")
    parser.add_argument("--config-output", default="best_config.json", help="Output file for best lens config")

    # Tuning
    parser.add_argument("--windows-per-config", type=int, default=WINDOWS_PER_CONFIG, help=f"Windows per config (default: {WINDOWS_PER_CONFIG})")
    parser.add_argument("--probe-timeout", type=int, default=PROBE_TIMEOUT_SEC, help=f"Probe warm-up timeout in seconds (default: {PROBE_TIMEOUT_SEC})")

    args = parser.parse_args()

    if len(args.n_shot_files) != len(args.class_names):
        parser.error("Number of --n-shot-files must match --class-names")

    WINDOWS_PER_CONFIG = args.windows_per_config
    PROBE_TIMEOUT_SEC = args.probe_timeout

    run_optimizer(args)


if __name__ == "__main__":
    main()
