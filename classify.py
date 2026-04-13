#!/usr/bin/env python3
"""
Run a Machine State Lens session with an optimized config.

Takes the `best_config.json` produced by optimize.py (or any compatible
lens config) and streams an inference CSV through Newton, writing the
per-window predictions to disk. Optionally evaluates against ground
truth if the inference file has a label column.

Examples:

    # Use the optimizer's winning config to classify a full well
    python classify.py \\
        --config-file best_config.json \\
        --n-shot-files data/nshot_drilling.csv data/nshot_not_drilling.csv \\
        --class-names drilling not_drilling \\
        --inference-file data/full_well.csv \\
        --output predictions.csv

    # Evaluate against ground truth too (--label-column)
    python classify.py \\
        --config-file best_config.json \\
        --n-shot-files data/nshot_drilling.csv data/nshot_not_drilling.csv \\
        --class-names drilling not_drilling \\
        --inference-file data/labeled_well.csv \\
        --label-column label \\
        --output predictions.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Reuse helpers from optimize.py to avoid duplication
from optimize import (
    SSEReader,
    clean_stale_lenses,
    compute_f1,
    create_session,
    destroy_session,
    get_window_ground_truth,
    load_csv,
    stream_window,
    transpose_window,
    upload_file,
    DEFAULT_ENDPOINT,
)


def load_lens_config(path):
    """Read best_config.json and pull the bits we need to recreate the session."""
    with open(path) as f:
        cfg = json.load(f)
    csv_cfg = cfg.get("csv_configs", {})
    knn_cfg = cfg.get("knn_configs", {})
    return {
        "window_size": csv_cfg.get("window_size", 64),
        "step_size": csv_cfg.get("step_size", csv_cfg.get("window_size", 64)),
        "data_columns": csv_cfg.get("data_columns", []),
        "timestamp_column": csv_cfg.get("timestamp_column", "timestamp"),
        "n_neighbors": knn_cfg.get("n_neighbors", 5),
        "metric": knn_cfg.get("metric", "euclidean"),
        "weights": knn_cfg.get("weights", "uniform"),
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-file", required=True, help="Lens config JSON (best_config.json from optimize.py)")
    p.add_argument("--n-shot-files", nargs="+", required=True, help="One CSV per class")
    p.add_argument("--class-names", nargs="+", required=True, help="Class names matching n-shot files")
    p.add_argument("--inference-file", required=True, help="CSV to classify (sensor data, optional label column)")
    p.add_argument("--data-columns", nargs="+", help="Override data columns (default: from config file)")
    p.add_argument("--timestamp-column", help="Override timestamp column (default: from config file)")
    p.add_argument("--label-column", help="Optional ground-truth column for evaluation")
    p.add_argument("--output", default="predictions.csv", help="Output CSV path (default: predictions.csv)")
    p.add_argument("--api-key", help="Archetype AI API key (or set ATAI_API_KEY env var)")
    p.add_argument("--api-endpoint", default=DEFAULT_ENDPOINT, help=f"API endpoint (default: {DEFAULT_ENDPOINT})")
    p.add_argument("--probe-timeout", type=int, default=90, help="Probe warm-up timeout seconds (default: 90)")
    p.add_argument("--stream-delay", type=float, default=0.5, help="Delay between window sends (default: 0.5s)")
    p.add_argument("--max-windows", type=int, help="Stop after this many windows (default: process whole file)")
    args = p.parse_args()

    if len(args.n_shot_files) != len(args.class_names):
        p.error("Number of --n-shot-files must match --class-names")

    api_key = args.api_key or os.getenv("ATAI_API_KEY")
    if not api_key:
        sys.exit("Error: ATAI_API_KEY not set. Use --api-key or set the environment variable.")

    cfg = load_lens_config(args.config_file)
    data_columns = args.data_columns or cfg["data_columns"]
    timestamp_column = args.timestamp_column or cfg["timestamp_column"]
    if not data_columns:
        sys.exit("Error: no data columns found in config file. Pass --data-columns explicitly.")

    print(f"Config: window={cfg['window_size']} step={cfg['step_size']} "
          f"k={cfg['n_neighbors']} metric={cfg['metric']} weights={cfg['weights']}")
    print(f"Data columns: {data_columns}")
    print(f"Timestamp column: {timestamp_column}")

    # Load inference data
    print(f"\nLoading inference data: {args.inference_file}")
    rows = load_csv(args.inference_file, data_columns, args.label_column)
    print(f"  {len(rows):,} rows")

    window_size = cfg["window_size"]
    step_size = cfg["step_size"]
    max_windows = args.max_windows or (len(rows) // step_size)
    print(f"  Will stream up to {max_windows:,} windows")

    # Upload n-shot files
    print("\nUploading n-shot files...")
    n_shot_map = {}
    for file_path, class_name in zip(args.n_shot_files, args.class_names):
        print(f"  {file_path} → {class_name}")
        file_id = upload_file(args.api_endpoint, api_key, file_path)
        n_shot_map[class_name] = file_id
        print(f"    file_id: {file_id}")

    clean_stale_lenses(args.api_endpoint, api_key)

    # Build session config in the shape create_session expects
    session_config = {
        "window_size": window_size,
        "n_neighbors": cfg["n_neighbors"],
        "metric": cfg["metric"],
        "weights": cfg["weights"],
    }

    print("\nCreating session...")
    reader = None
    session_id = None
    try:
        session_id, sse_url = create_session(
            args.api_endpoint, api_key, n_shot_map, session_config,
            data_columns, timestamp_column,
        )

        reader = SSEReader(sse_url, api_key)
        reader.start()

        # Probe + warm-up
        print("Warming up (probe)...")
        probe_data = transpose_window(rows, 0, window_size, data_columns)
        stream_window(args.api_endpoint, api_key, session_id, probe_data, 0)
        first = reader.wait_for_first(args.probe_timeout)
        if first is None:
            print(f"  Warm-up timed out after {args.probe_timeout}s; continuing anyway")
        else:
            print(f"  Warm-up complete: {first}")
            reader.drain()

        # Stream windows
        print(f"\nStreaming up to {max_windows:,} windows (step={step_size})...")
        predictions = []
        ground_truths = []
        timestamps = []
        ts_queue = []
        gt_queue = []

        def absorb():
            for result in reader.drain():
                if not ts_queue:
                    break
                predictions.append(result)
                timestamps.append(ts_queue.pop(0))
                if gt_queue:
                    ground_truths.append(gt_queue.pop(0))

        for i in range(1, max_windows + 1):
            start = i * step_size
            if start + window_size > len(rows):
                break

            ts = rows[start].get(timestamp_column, "") if timestamp_column else ""
            ts_queue.append(ts)
            if args.label_column:
                gt_queue.append(
                    get_window_ground_truth(rows, start, window_size, args.label_column, args.class_names)
                )

            sensor_data = transpose_window(rows, start, window_size, data_columns)
            stream_window(args.api_endpoint, api_key, session_id, sensor_data, i)
            time.sleep(args.stream_delay)
            absorb()

            if i % 25 == 0:
                print(f"  streamed {i:,} / {max_windows:,} windows ({len(predictions):,} predictions back)")

        # Drain trailing results. Cap scales with window_size since Newton
        # takes longer to fire predictions at larger windows. Bail early on
        # 5 consecutive silent seconds.
        max_drain_sec = max(20, window_size)
        no_new_streak = 0
        prev_count = len(predictions)
        deadline = time.time() + max_drain_sec
        while time.time() < deadline and ts_queue:
            time.sleep(1)
            absorb()
            if len(predictions) == prev_count:
                no_new_streak += 1
                if no_new_streak >= 5:
                    break
            else:
                no_new_streak = 0
                prev_count = len(predictions)

        print(f"\nReceived {len(predictions):,} predictions for {max_windows:,} streamed windows.")

        # Write output
        print(f"\nWriting {args.output}...")
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            header = ["window_index"]
            if timestamp_column:
                header.append(timestamp_column)
            header.append("prediction")
            if args.label_column:
                header.append("ground_truth")
            w.writerow(header)
            for i, pred in enumerate(predictions):
                row = [i]
                if timestamp_column:
                    row.append(timestamps[i] if i < len(timestamps) else "")
                row.append(pred)
                if args.label_column and i < len(ground_truths):
                    row.append(ground_truths[i] if ground_truths[i] is not None else "")
                w.writerow(row)
        print(f"  Wrote {len(predictions):,} rows")

        # Optional evaluation
        if args.label_column and ground_truths:
            scored = [(p, g) for p, g in zip(predictions, ground_truths) if g is not None]
            if scored:
                preds, gts = zip(*scored)
                metrics = compute_f1(list(preds), list(gts), args.class_names)
                print(f"\nEvaluation ({len(scored):,} unanimous-window pairs):")
                print(f"  macro F1: {metrics['macro_f1']*100:.1f}%")
                print(f"  accuracy: {metrics['accuracy']*100:.1f}%")
                for cls, stats in metrics["per_class"].items():
                    print(f"  {cls}: P={stats['precision']*100:.1f}% R={stats['recall']*100:.1f}% "
                          f"F1={stats['f1']*100:.1f}% (TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})")

    finally:
        if reader is not None:
            reader.stop()
        if session_id is not None:
            destroy_session(args.api_endpoint, api_key, session_id)


if __name__ == "__main__":
    main()
