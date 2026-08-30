#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Convert VisDrone2019-MOT splits to the COCO-video JSON used by DAMOT.

Images are not copied. The generated JSON stores absolute image paths so the
original VisDrone folders can remain in place.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2


VISDRONE_TO_COCO = {1: 1, 4: 2, 5: 3, 6: 4, 9: 5}
CATEGORIES = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "van"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bus"},
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    autodl_root = Path(os.environ.get("AUTODL_ROOT", "/root/autodl-tmp"))
    parser = argparse.ArgumentParser(
        description="Prepare VisDrone2019-MOT COCO annotations without copying images."
    )
    parser.add_argument(
        "--train-root",
        default=str(autodl_root / "data" / "VisDrone2019-MOT-train"),
    )
    parser.add_argument(
        "--val-root",
        default=str(autodl_root / "data" / "VisDrone2019-MOT-val"),
    )
    parser.add_argument(
        "--test-dev-root",
        default=str(autodl_root / "data" / "VisDrone2019-MOT-test-dev"),
    )
    parser.add_argument(
        "--output-root", default=str(repo_root / "datasets" / "visdrone")
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test-dev"),
        default=("train", "val", "test-dev"),
    )
    return parser.parse_args()


def resolve_dataset_root(path):
    root = Path(path).expanduser().resolve()
    if (root / "sequences").is_dir() and (root / "annotations").is_dir():
        return root
    nested = root / root.name
    if (nested / "sequences").is_dir() and (nested / "annotations").is_dir():
        return nested
    raise FileNotFoundError(
        "VisDrone split root must contain sequences/ and annotations/: {}".format(root)
    )


def load_annotations(annotation_file):
    by_frame = defaultdict(list)
    if not annotation_file.is_file():
        return by_frame
    with annotation_file.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            fields = line.split(",")
            if len(fields) < 8:
                raise ValueError(
                    "{}:{} has fewer than 8 columns".format(
                        annotation_file, line_number
                    )
                )
            values = [float(value) for value in fields[:10]]
            frame_id = int(values[0])
            by_frame[frame_id].append(values)
    return by_frame


def image_size(image_file):
    image = cv2.imread(str(image_file))
    if image is None:
        raise RuntimeError("Cannot read image: {}".format(image_file))
    height, width = image.shape[:2]
    return width, height


def convert_split(source_root, output_file):
    source_root = resolve_dataset_root(source_root)
    sequence_root = source_root / "sequences"
    annotation_root = source_root / "annotations"

    images = []
    annotations = []
    videos = []
    image_id = 1
    annotation_id = 1

    sequence_dirs = sorted(path for path in sequence_root.iterdir() if path.is_dir())
    for video_id, sequence_dir in enumerate(sequence_dirs, 1):
        sequence_name = sequence_dir.name
        videos.append({"id": video_id, "file_name": sequence_name})
        frame_annotations = load_annotations(
            annotation_root / (sequence_name + ".txt")
        )
        frame_files = sorted(
            sequence_dir.glob("*.jpg"), key=lambda path: int(path.stem)
        )
        if not frame_files:
            raise RuntimeError("No JPG frames found in {}".format(sequence_dir))
        width, height = image_size(frame_files[0])

        for frame_file in frame_files:
            frame_id = int(frame_file.stem)
            current_image_id = image_id
            images.append(
                {
                    "id": current_image_id,
                    "file_name": frame_file.resolve().as_posix(),
                    "frame_id": frame_id,
                    "prev_image_id": current_image_id - 1 if frame_id > 1 else -1,
                    "next_image_id": (
                        current_image_id + 1 if frame_id < len(frame_files) else -1
                    ),
                    "video_id": video_id,
                    "height": height,
                    "width": width,
                }
            )

            for row in frame_annotations.get(frame_id, ()):
                track_id = int(row[1])
                left, top, box_width, box_height = row[2:6]
                score = int(row[6])
                original_category = int(row[7])
                if (
                    score != 1
                    or original_category not in VISDRONE_TO_COCO
                    or box_width <= 0
                    or box_height <= 0
                ):
                    continue
                left = max(0.0, left)
                top = max(0.0, top)
                box_width = min(box_width, width - left)
                box_height = min(box_height, height - top)
                if box_width <= 0 or box_height <= 0:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "category_id": VISDRONE_TO_COCO[original_category],
                        "image_id": current_image_id,
                        "track_id": track_id,
                        "bbox": [left, top, box_width, box_height],
                        "area": box_width * box_height,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
            image_id += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "annotations": annotations,
        "videos": videos,
        "categories": CATEGORIES,
    }
    with output_file.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
    print(
        "{}: {} videos, {} images, {} target boxes -> {}".format(
            source_root.name,
            len(videos),
            len(images),
            len(annotations),
            output_file,
        )
    )


def main():
    args = parse_args()
    roots = {
        "train": args.train_root,
        "val": args.val_root,
        "test-dev": args.test_dev_root,
    }
    annotation_dir = Path(args.output_root).resolve() / "annotations"
    for split in args.splits:
        convert_split(roots[split], annotation_dir / (split + ".json"))


if __name__ == "__main__":
    main()
