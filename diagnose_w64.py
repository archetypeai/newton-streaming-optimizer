#!/usr/bin/env python3
"""
Diagnostic: why do w64+ configs silently return no predictions?

Runs three focused experiments on a single w64 config (k=3, euclidean, uniform)
to test distinct hypotheses. Each creates its own lens session. Prints raw
per-second counts of predictions arriving so we can see timing, not just
aggregate results.

Usage:
    python diagnose_w64.py \\
        --inference-file examples/drilling/inference.csv \\
        --n-shot-files examples/drilling/nshot_drilling.csv examples/drilling/nshot_not_drilling.csv \\
        --class-names drilling not_drilling \\
        --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \\
        --timestamp-column DATE_TIME
"""

import argparse
import json
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

from optimize import (
    SSEReader,
    api_post,
    api_url,
    clean_stale_lenses,
    find_mixed_offset,
    load_csv,
    stream_window,
    transpose_window,
    upload_file,
    DEFAULT_ENDPOINT,
)


def create_session_custom(endpoint, api_key, n_shot_map, window_size, buffer_size,
                          data_columns, timestamp_column,
                          n_neighbors=3, metric="euclidean", weights="uniform"):
    """Create a lens session with explicit buffer_size (not necessarily = window_size)."""
    lens_name = f"diagnose-w64-{int(time.time())}"
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
                "buffer_size": buffer_size,
                "input_n_shot": n_shot_map,
                "csv_configs": {
                    "timestamp_column": timestamp_column,
                    "data_columns": data_columns,
                    "window_size": window_size,
                    "step_size": window_size,
                },
                "knn_configs": {
                    "n_neighbors": n_neighbors,
                    "metric": metric,
                    "weights": weights,
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
    t0 = time.time()
    while time.time() - t0 < 60:
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


def destroy(endpoint, api_key, session_id):
    try:
        api_post(endpoint, api_key, "/lens/sessions/destroy", {"session_id": session_id}, timeout=15)
    except Exception:
        pass


def run_experiment(name, endpoint, api_key, rows, offset, data_columns, timestamp_column,
                   n_shot_map, window_size, buffer_size, num_windows, stream_delay,
                   post_stream_wait_sec):
    """Run one experiment variant, print per-second prediction counts."""
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  window={window_size} buffer={buffer_size} stream_windows={num_windows} "
          f"delay={stream_delay}s post_stream_wait={post_stream_wait_sec}s")
    print("=" * 70)

    session_id = None
    reader = None
    try:
        print("  Creating session...")
        t_create = time.time()
        session_id, sse_url = create_session_custom(
            endpoint, api_key, n_shot_map, window_size, buffer_size,
            data_columns, timestamp_column,
        )
        print(f"  Session ready in {time.time() - t_create:.1f}s: {session_id}")

        reader = SSEReader(sse_url, api_key)
        reader.start()
        # Give the SSE GET a chance to actually connect before we stream data
        time.sleep(2)

        # Stream inference windows; report per-second prediction counts
        t_stream_start = time.time()
        total_received = 0
        last_report_sec = 0

        for i in range(num_windows):
            start = offset + i * window_size
            if start + window_size > len(rows):
                break
            sensor = transpose_window(rows, start, window_size, data_columns)
            try:
                stream_window(endpoint, api_key, session_id, sensor, i)
            except Exception as e:
                print(f"  stream_window error at i={i}: {e}")
                break
            time.sleep(stream_delay)

            # Report every ~10s during streaming
            elapsed = int(time.time() - t_stream_start)
            if elapsed >= last_report_sec + 10:
                new = len(reader.drain())  # read and discard; we only care about counts
                total_received += new
                print(f"  t+{elapsed:3d}s streaming: sent {i + 1} windows, "
                      f"received {new} new preds this interval (total {total_received})")
                last_report_sec = elapsed

        t_stream_end = time.time()
        stream_elapsed = t_stream_end - t_stream_start
        # Final drain of anything that arrived during last interval
        new = len(reader.drain())
        total_received += new
        print(f"  Streaming complete: {stream_elapsed:.1f}s, {total_received} preds received so far")

        # Post-stream wait: do results arrive after we stop sending?
        print(f"  Waiting {post_stream_wait_sec}s for trailing results...")
        t_wait_start = time.time()
        last_report_sec = 0
        while time.time() - t_wait_start < post_stream_wait_sec:
            time.sleep(1)
            new = len(reader.drain())
            if new > 0:
                total_received += new
                print(f"  t+{int(time.time() - t_wait_start)}s post-stream: +{new} preds "
                      f"(total {total_received})")
                last_report_sec = int(time.time() - t_wait_start)

        print(f"\n  RESULT: {total_received} total predictions received")
        return total_received

    finally:
        if reader is not None:
            reader.stop()
        if session_id is not None:
            destroy(endpoint, api_key, session_id)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inference-file", required=True)
    p.add_argument("--n-shot-files", nargs="+", required=True)
    p.add_argument("--class-names", nargs="+", required=True)
    p.add_argument("--data-columns", nargs="+", required=True)
    p.add_argument("--timestamp-column", default="timestamp")
    p.add_argument("--label-column", help="Optional; used only for mixed-offset search")
    p.add_argument("--api-key", help="Or set ATAI_API_KEY env var")
    p.add_argument("--api-endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--window-size", type=int, default=64)
    args = p.parse_args()

    api_key = args.api_key or os.getenv("ATAI_API_KEY")
    if not api_key:
        sys.exit("Error: ATAI_API_KEY not set")

    print(f"Loading inference data: {args.inference_file}")
    rows = load_csv(args.inference_file, args.data_columns, args.label_column)
    print(f"  {len(rows):,} rows")

    print("Uploading n-shot files...")
    n_shot_map = {}
    for file_path, class_name in zip(args.n_shot_files, args.class_names):
        print(f"  {file_path} → {class_name}")
        file_id = upload_file(args.api_endpoint, api_key, file_path)
        n_shot_map[class_name] = file_id

    clean_stale_lenses(args.api_endpoint, api_key)

    # Pick the mixed-offset section so ground truth is balanced (though we don't eval here)
    if args.label_column:
        offset, mix = find_mixed_offset(rows, args.label_column, args.class_names, args.window_size)
        print(f"Using data offset {offset} (mix {mix*100:.1f}%)")
    else:
        offset = 0
        print(f"Using data offset 0 (no label column for balance search)")

    # Baseline ------------------------------------------------------------
    # Same as optimize.py: 100 windows × 1s, but with a longer post-stream wait.
    r1 = run_experiment(
        "1) Baseline + longer wait (300s post-stream, no silence bail)",
        args.api_endpoint, api_key, rows, offset, args.data_columns, args.timestamp_column,
        n_shot_map,
        window_size=args.window_size,
        buffer_size=args.window_size,  # matches optimize.py default
        num_windows=100,
        stream_delay=1.0,
        post_stream_wait_sec=300,
    )

    # Larger buffer ------------------------------------------------------
    # Does Newton need more buffered samples before firing predictions?
    r2 = run_experiment(
        "2) Larger buffer (2× window_size)",
        args.api_endpoint, api_key, rows, offset, args.data_columns, args.timestamp_column,
        n_shot_map,
        window_size=args.window_size,
        buffer_size=args.window_size * 2,
        num_windows=100,
        stream_delay=1.0,
        post_stream_wait_sec=180,
    )

    # Slower pacing ------------------------------------------------------
    # Is Newton back-pressured by our 1s-between-windows stream?
    r3 = run_experiment(
        "3) Slower pacing (3s between windows)",
        args.api_endpoint, api_key, rows, offset, args.data_columns, args.timestamp_column,
        n_shot_map,
        window_size=args.window_size,
        buffer_size=args.window_size,
        num_windows=100,
        stream_delay=3.0,
        post_stream_wait_sec=180,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  1) Baseline + longer wait:       {r1} predictions")
    print(f"  2) Larger buffer (2× window):    {r2} predictions")
    print(f"  3) Slower pacing (3s delay):     {r3} predictions")
    if max(r1, r2, r3) == 0:
        print("\n  All three variations returned zero predictions. w64 is likely")
        print("  unsupported at the encoder/lens level on this API deployment.")
    else:
        best = max([("baseline+longer wait", r1), ("larger buffer", r2), ("slower pacing", r3)],
                   key=lambda t: t[1])
        print(f"\n  Best variation: {best[0]} ({best[1]} predictions)")


if __name__ == "__main__":
    main()
