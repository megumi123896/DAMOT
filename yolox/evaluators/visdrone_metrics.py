#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""VisDrone2019-MOT aggregate metrics with official ignore-region filtering."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np


TARGET_CATEGORIES = {1, 4, 5, 6, 9}
IGNORE_CATEGORIES = {0, 11}
METRICS = [
    "mota",
    "idf1",
    "num_switches",
    "num_false_positives",
    "num_misses",
    "mostly_tracked",
    "mostly_lost",
]


def resolve_visdrone_root(path):
    root = Path(path).expanduser().resolve()
    if (root / "sequences").is_dir() and (root / "annotations").is_dir():
        return root
    nested = root / root.name
    if (nested / "sequences").is_dir() and (nested / "annotations").is_dir():
        return nested
    raise FileNotFoundError(
        "VisDrone root must contain sequences/ and annotations/: {}".format(root)
    )


def _read_rows(path):
    rows = []
    if not path.is_file() or path.stat().st_size == 0:
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row:
                continue
            try:
                values = [float(value) for value in row[:10]]
            except ValueError as error:
                raise ValueError("Invalid row {}:{}".format(path, line_number)) from error
            if len(values) < 6:
                raise ValueError("Too few columns at {}:{}".format(path, line_number))
            rows.append(values)
    return rows


def _union_intersection_area(box, regions):
    left, top, width, height = box
    right = left + width
    bottom = top + height
    clipped = []
    for region in regions:
        r_left, r_top, r_width, r_height = region
        r_right = r_left + r_width
        r_bottom = r_top + r_height
        x1 = max(left, r_left)
        y1 = max(top, r_top)
        x2 = min(right, r_right)
        y2 = min(bottom, r_bottom)
        if x2 > x1 and y2 > y1:
            clipped.append((x1, y1, x2, y2))
    if not clipped:
        return 0.0

    x_edges = sorted({edge for rect in clipped for edge in (rect[0], rect[2])})
    area = 0.0
    for x1, x2 in zip(x_edges[:-1], x_edges[1:]):
        if x2 <= x1:
            continue
        intervals = sorted(
            (rect[1], rect[3])
            for rect in clipped
            if rect[0] < x2 and rect[2] > x1
        )
        if not intervals:
            continue
        merged_height = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                merged_height += end - start
                start, end = next_start, next_end
        merged_height += end - start
        area += (x2 - x1) * merged_height
    return area


def _outside_ignore_regions(box, ignore_regions, threshold=0.5):
    box_area = max(box[2], 0.0) * max(box[3], 0.0)
    if box_area <= 0:
        return False
    ignored_area = _union_intersection_area(box, ignore_regions)
    return ignored_area / box_area < threshold


def _load_ground_truth(annotation_file):
    rows = _read_rows(annotation_file)
    ignore_by_frame = defaultdict(list)
    candidates = []
    for row in rows:
        frame_id = int(row[0])
        category = int(row[7]) if len(row) > 7 else -1
        box = row[2:6]
        if category in IGNORE_CATEGORIES:
            ignore_by_frame[frame_id].append(box)
        if int(row[6]) == 1 and category in TARGET_CATEGORIES:
            candidates.append(row)

    # The official toolkit drops GT and tracker boxes covered >=50% by ignored regions.
    candidates = [
        row
        for row in candidates
        if _outside_ignore_regions(
            row[2:6], ignore_by_frame.get(int(row[0]), ())
        )
    ]

    # Equivalent to breakGts.m: one identity cannot span multiple categories.
    identity_map = {}
    gt_by_frame = defaultdict(list)
    next_identity = 1
    for row in candidates:
        key = (int(row[1]), int(row[7]))
        if key not in identity_map:
            identity_map[key] = next_identity
            next_identity += 1
        gt_by_frame[int(row[0])].append(
            (identity_map[key], np.asarray(row[2:6], dtype=float))
        )
    return gt_by_frame, ignore_by_frame


def _load_tracker_results(result_file, ignore_by_frame):
    tracker_by_frame = defaultdict(list)
    for row in _read_rows(result_file):
        frame_id = int(row[0])
        track_id = int(row[1])
        box = row[2:6]
        if track_id < 0 or not _outside_ignore_regions(
            box, ignore_by_frame.get(frame_id, ())
        ):
            continue
        tracker_by_frame[frame_id].append(
            (track_id, np.asarray(box, dtype=float))
        )
    return tracker_by_frame


def _build_accumulator(gt_by_frame, tracker_by_frame, frame_count):
    accumulator = mm.MOTAccumulator(auto_id=True)
    for frame_id in range(1, frame_count + 1):
        gt_rows = gt_by_frame.get(frame_id, ())
        tracker_rows = tracker_by_frame.get(frame_id, ())
        gt_ids = [row[0] for row in gt_rows]
        tracker_ids = [row[0] for row in tracker_rows]
        gt_boxes = np.asarray([row[1] for row in gt_rows], dtype=float).reshape(-1, 4)
        tracker_boxes = np.asarray(
            [row[1] for row in tracker_rows], dtype=float
        ).reshape(-1, 4)
        distances = mm.distances.iou_matrix(
            gt_boxes, tracker_boxes, max_iou=0.5
        )
        accumulator.update(gt_ids, tracker_ids, distances)
    return accumulator


def _display_summary(summary):
    display = summary.copy()
    display["mota"] *= 100.0
    display["idf1"] *= 100.0
    display = display.rename(
        columns={
            "mota": "MOTA↑",
            "idf1": "IDF1↑",
            "num_switches": "IDs↓",
            "num_false_positives": "FP↓",
            "num_misses": "FN↓",
            "mostly_tracked": "MT↑",
            "mostly_lost": "ML↓",
        }
    )
    formatters = {
        "MOTA↑": lambda value: "{:.2f}".format(value),
        "IDF1↑": lambda value: "{:.2f}".format(value),
    }
    for name in ("IDs↓", "FP↓", "FN↓", "MT↑", "ML↓"):
        formatters[name] = lambda value: "{}".format(int(round(value)))
    print(display.to_string(formatters=formatters))


def evaluate_visdrone_mot(dataset_root, results_dir, output_dir=None):
    dataset_root = resolve_visdrone_root(dataset_root)
    results_dir = Path(results_dir).expanduser().resolve()
    accumulators = []
    names = []

    sequence_dirs = sorted(
        path for path in (dataset_root / "sequences").iterdir() if path.is_dir()
    )
    for sequence_dir in sequence_dirs:
        sequence_name = sequence_dir.name
        annotation_file = dataset_root / "annotations" / (sequence_name + ".txt")
        result_file = results_dir / (sequence_name + ".txt")
        gt_by_frame, ignore_by_frame = _load_ground_truth(annotation_file)
        tracker_by_frame = _load_tracker_results(result_file, ignore_by_frame)
        frame_count = len(list(sequence_dir.glob("*.jpg")))
        accumulators.append(
            _build_accumulator(gt_by_frame, tracker_by_frame, frame_count)
        )
        names.append(sequence_name)

    if not accumulators:
        raise RuntimeError("No VisDrone sequences found in {}".format(dataset_root))

    metric_host = mm.metrics.create()
    summary = metric_host.compute_many(
        accumulators,
        names=names,
        metrics=METRICS,
        generate_overall=True,
    )
    _display_summary(summary)

    if output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "mot_metrics.csv")
        serializable = {}
        for sequence_name, row in summary.iterrows():
            serializable[sequence_name] = {
                key: (None if not math.isfinite(float(value)) else float(value))
                for key, value in row.items()
            }
        with (output_dir / "mot_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=2, ensure_ascii=False)
    return summary
