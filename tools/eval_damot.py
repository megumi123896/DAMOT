#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Run DAMOT on VisDrone2019-MOT and report the requested MOT metrics."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from yolox.evaluators import MOTEvaluator
from yolox.evaluators.visdrone_metrics import evaluate_visdrone_mot
from yolox.exp import get_exp
from yolox.utils import fuse_model, setup_logger


def parse_args():
    autodl_root = Path(os.environ.get("AUTODL_ROOT", "/root/autodl-tmp"))
    parser = argparse.ArgumentParser(
        description="Evaluate DAMOT on VisDrone2019-MOT val or test-dev."
    )
    parser.add_argument(
        "-f",
        "--exp-file",
        default=str(REPO_ROOT / "exps" / "example" / "mot" / "damot_visdrone.py"),
    )
    parser.add_argument("-c", "--ckpt", required=True)
    parser.add_argument("--split", choices=("val", "test-dev"), default="val")
    parser.add_argument(
        "--val-root",
        default=str(autodl_root / "data" / "VisDrone2019-MOT-val"),
    )
    parser.add_argument(
        "--test-dev-root",
        default=str(autodl_root / "data" / "VisDrone2019-MOT-test-dev"),
    )
    parser.add_argument(
        "--coco-root", default=str(REPO_ROOT / "datasets" / "visdrone")
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--nms", type=float, default=0.7)
    parser.add_argument("--track-thresh", type=float, default=0.6)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--min-box-area", type=float, default=100.0)
    parser.add_argument("--reid-batch-size", type=int, default=128)
    parser.add_argument(
        "--config-file",
        default=str(
            REPO_ROOT
            / "fast_reid"
            / "configs"
            / "VisDrone"
            / "damot_veriwild_R50-ibn.yml"
        ),
    )
    parser.add_argument(
        "--reid-weights",
        default=str(REPO_ROOT / "pretrained" / "veriwild_bot_R50-ibn.pth"),
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning("Missing checkpoint keys: {}", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning("Unexpected checkpoint keys: {}", incompatible.unexpected_keys)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DAMOT evaluation.")
    torch.cuda.set_device(args.device)
    cudnn.benchmark = True

    args.name = "damot-visdrone"
    args.opts = []
    args.match_thresh = 0.9
    args.mot20 = False

    os.environ["VISDRONE_COCO_ROOT"] = str(Path(args.coco_root).resolve())
    exp = get_exp(args.exp_file, None)
    exp.data_dir = str(Path(args.coco_root).resolve())
    exp.val_ann = args.split + ".json"
    exp.val_name = ""
    exp.test_conf = args.conf
    exp.nmsthre = args.nms

    annotation_file = Path(exp.data_dir) / "annotations" / exp.val_ann
    if not annotation_file.is_file():
        raise FileNotFoundError(
            "Missing {}. Run tools/prepare_visdrone.py first.".format(annotation_file)
        )
    if not Path(args.reid_weights).is_file():
        raise FileNotFoundError("Missing ReID weights: {}".format(args.reid_weights))

    checkpoint_file = Path(args.ckpt).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "YOLOX_outputs" / "damot_visdrone_{}".format(args.split)
    )
    results_dir = output_dir / "track_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(str(output_dir), distributed_rank=0, filename="eval_log.txt", mode="a")

    model = exp.get_model()
    load_checkpoint(model, checkpoint_file)
    model.cuda(args.device).eval()
    if args.fuse:
        model = fuse_model(model)

    val_loader = exp.get_eval_loader(batch_size=1, is_distributed=False)
    evaluator = MOTEvaluator(
        args=args,
        dataloader=val_loader,
        img_size=exp.test_size,
        confthre=exp.test_conf,
        nmsthre=exp.nmsthre,
        num_classes=exp.num_classes,
    )
    _, _, detection_summary = evaluator.evaluate_damot(
        model,
        distributed=False,
        half=args.fp16,
        trt_file=None,
        decoder=None,
        test_size=exp.test_size,
        result_folder=str(results_dir),
    )
    logger.info("Detection evaluation:\n{}", detection_summary)

    dataset_root = args.val_root if args.split == "val" else args.test_dev_root
    evaluate_visdrone_mot(
        dataset_root=dataset_root,
        results_dir=results_dir,
        output_dir=output_dir,
    )
    logger.info("Tracking results and metrics saved to {}", output_dir)


if __name__ == "__main__":
    main()
