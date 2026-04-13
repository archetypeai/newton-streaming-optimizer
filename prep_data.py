#!/usr/bin/env python3
"""
Data prep for Newton Streaming Optimizer.

Splits a labeled time-series CSV into the inputs that optimize.py expects:
  - One n-shot CSV per class (sensor data only)
  - One inference CSV (sensor data + label column for ground-truth eval)

Picks the longest contiguous run of each class for n-shot examples, and
finds the most class-balanced contiguous slice for the inference file.

Example (Volve drilling data):
    python prep_data.py \
        --input-file /path/to/volve_raw_labeled.csv \
        --output-dir data/volve \
        --data-columns BPOS DBTM FLWI HDTH HKLD ROP RPM SPPA WOB \
        --timestamp-column DATE_TIME \
        --label-column label

Example with raw codes (e.g. ACTC) needing remapping:
    python prep_data.py \
        --input-file raw.csv \
        --output-dir data/prepared \
        --data-columns sensor_1 sensor_2 sensor_3 \
        --label-column ACTC \
        --label-mapping "1:DRILLING,2:DRILLING,3:NOT_DRILLING,4:NOT_DRILLING"
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_label_mapping(spec):
    if not spec:
        return None
    out = {}
    for pair in spec.split(","):
        if ":" not in pair:
            sys.exit(f"Invalid --label-mapping entry '{pair}' (expected 'raw:class')")
        raw, cls = pair.split(":", 1)
        out[raw.strip()] = cls.strip()
    return out


def safe_filename(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).lower()


def scan_labels(input_file, label_column, label_map, allowed_classes):
    """Single pass over the file. Returns interned class per row (or None to skip)."""
    labels = []
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        if label_column not in reader.fieldnames:
            sys.exit(f"--label-column '{label_column}' not found. Available: {reader.fieldnames}")
        for row in reader:
            raw = (row.get(label_column) or "").strip()
            cls = label_map.get(raw) if label_map else raw
            if not cls:
                labels.append(None)
                continue
            if allowed_classes and cls not in allowed_classes:
                labels.append(None)
                continue
            labels.append(sys.intern(cls))
    return labels


def longest_run(labels, target):
    best_start, best_len = 0, 0
    i, n = 0, len(labels)
    while i < n:
        if labels[i] == target:
            j = i
            while j < n and labels[j] == target:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start = i
            i = j
        else:
            i += 1
    return best_start, best_len


def find_balanced_slice(labels, classes, slice_size):
    """Sliding window over labels; returns offset of slice with highest min-class share."""
    n = len(labels)
    slice_size = min(slice_size, n)
    counter = Counter()
    for i in range(slice_size):
        if labels[i] is not None:
            counter[labels[i]] += 1

    def score(c):
        if any(c.get(cls, 0) == 0 for cls in classes):
            return -1.0
        total = sum(c.values())
        return min(c[cls] for cls in classes) / total if total else -1.0

    best_offset, best_score = 0, score(counter)
    for i in range(slice_size, n):
        out_lab = labels[i - slice_size]
        in_lab = labels[i]
        if out_lab is not None:
            counter[out_lab] -= 1
            if counter[out_lab] == 0:
                del counter[out_lab]
        if in_lab is not None:
            counter[in_lab] += 1
        s = score(counter)
        if s > best_score:
            best_score = s
            best_offset = i - slice_size + 1
    return best_offset, best_score, slice_size


def extract_rows(input_file, ranges, sensor_cols, ts_col, label_col, labels):
    """Single pass to write all output files. ranges: dict path -> (start, end, include_label)."""
    writers = {}
    files = {}
    for path, (_, _, include_label) in ranges.items():
        files[path] = open(path, "w", newline="")
        w = csv.writer(files[path])
        header = ([ts_col] if ts_col else []) + list(sensor_cols)
        if include_label:
            header.append(label_col)
        w.writerow(header)
        writers[path] = w

    try:
        with open(input_file, newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                row_vals_base = ([row.get(ts_col, "")] if ts_col else []) + [
                    row.get(c, "") for c in sensor_cols
                ]
                for path, (start, end, include_label) in ranges.items():
                    if start <= idx < end:
                        if include_label:
                            lab = labels[idx] or (row.get(label_col) or "").strip()
                            writers[path].writerow(row_vals_base + [lab])
                        else:
                            writers[path].writerow(row_vals_base)
    finally:
        for fp in files.values():
            fp.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-file", required=True, help="Source labeled CSV.")
    p.add_argument("--output-dir", default="prepared_data", help="Where to write outputs.")
    p.add_argument("--data-columns", nargs="+", required=True, help="Sensor column names.")
    p.add_argument("--timestamp-column", default=None, help="Optional timestamp column.")
    p.add_argument("--label-column", required=True, help="Ground-truth label column.")
    p.add_argument("--label-mapping", default=None,
                   help="Optional remap, e.g. '1:DRILLING,2:DRILLING,3:NOT_DRILLING'.")
    p.add_argument("--classes", nargs="+", default=None,
                   help="Limit to these class names (after mapping). Default: all classes found.")
    p.add_argument("--n-shot-size", type=int, default=2000, help="Rows per n-shot file.")
    p.add_argument("--inference-size", type=int, default=200000,
                   help="Rows in inference file (a contiguous balanced slice). "
                        "Default 200K gives ample headroom for window sizes up to 1024.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = parse_label_mapping(args.label_mapping)
    allowed = set(args.classes) if args.classes else None

    print(f"Scanning labels in {args.input_file}...")
    labels = scan_labels(args.input_file, args.label_column, label_map, allowed)
    total = len(labels)

    counts = Counter(lab for lab in labels if lab)
    if not counts:
        sys.exit("No usable rows found. Check --label-column / --label-mapping / --classes.")

    print(f"Loaded {total:,} rows. Class distribution:")
    for cls, c in counts.most_common():
        print(f"  {cls:<20} {c:>10,}  ({c / total * 100:5.1f}%)")

    classes = args.classes if args.classes else sorted(counts.keys())
    missing = [c for c in classes if counts.get(c, 0) == 0]
    if missing:
        sys.exit(f"No rows found for class(es): {missing}")

    # n-shot ranges
    nshot_ranges = {}
    nshot_paths = {}
    print("\nN-shot selection:")
    for cls in classes:
        start, run_len = longest_run(labels, cls)
        if run_len < args.n_shot_size:
            print(f"  WARNING: '{cls}' longest contiguous run is {run_len:,} rows "
                  f"(< requested {args.n_shot_size:,})")
        n = min(args.n_shot_size, run_len)
        offset = start + (run_len - n) // 2
        path = out_dir / f"nshot_{safe_filename(cls)}.csv"
        nshot_paths[cls] = path
        nshot_ranges[path] = (offset, offset + n, False)
        print(f"  {cls:<20} rows {offset:,}-{offset + n:,}  (longest run: {run_len:,})")

    # Inference slice
    print("\nInference slice search...")
    inf_offset, inf_score, inf_size = find_balanced_slice(labels, classes, args.inference_size)
    inf_path = out_dir / "inference.csv"
    print(f"  rows {inf_offset:,}-{inf_offset + inf_size:,}  (min-class share: {inf_score:.1%})")

    # Per-class breakdown of the chosen slice
    slice_counts = Counter(labels[i] for i in range(inf_offset, inf_offset + inf_size) if labels[i])
    for cls in classes:
        c = slice_counts.get(cls, 0)
        print(f"    {cls:<20} {c:>10,}  ({c / inf_size * 100:5.1f}%)")

    # Extract all outputs in one streaming pass
    ranges = dict(nshot_ranges)
    ranges[inf_path] = (inf_offset, inf_offset + inf_size, True)

    print("\nWriting output files...")
    extract_rows(args.input_file, ranges, args.data_columns,
                 args.timestamp_column, args.label_column, labels)

    print(f"\nWrote:")
    print(f"  {inf_path}")
    for cls, p in nshot_paths.items():
        print(f"  {p}")

    # Suggested optimize.py command
    print("\nNext step:")
    cmd = ["python optimize.py",
           f"    --inference-file {inf_path}",
           f"    --n-shot-files " + " ".join(str(nshot_paths[c]) for c in classes),
           f"    --class-names " + " ".join(classes),
           f"    --data-columns " + " ".join(args.data_columns),
           f"    --label-column {args.label_column}"]
    if args.timestamp_column:
        cmd.append(f"    --timestamp-column {args.timestamp_column}")
    print(" \\\n".join(cmd))


if __name__ == "__main__":
    main()
