# DAMOT

**Difference-Aware Tiny-Object Enhancement for UAV Multi-Object Tracking**

DAMOT is a tracking-by-detection framework for UAV videos with dense tiny
objects, weak visual evidence, low-confidence detections, and strong camera
motion. It combines a YOLOX-X detector with a training-only difference-aware
enhancement module and UAV-aware trajectory association.

## Method

### Training-time Difference Enhancement Module (TDEM)

TDEM operates only on the highest-resolution PAFPN feature `P3` during
training:

```text
P3 -> Bilinear Reconstruction Module (BRM) -> reconstructed RGB image
   -> RGB reconstruction residual
   -> target-protected difference-domain background smoothing
   -> binary spatial prior + channel attention
   -> enhanced P3
```

BRM restores stride-8 `P3` to image resolution with three bilinear 2x
upsampling blocks. Each block uses bilinear interpolation, a 3x3 channel
reduction convolution, and two 3x3 convolution-ReLU stages. The final 3x3
convolution maps the feature to three RGB channels.

Inside ground-truth boxes, the original RGB reconstruction residual is
preserved. Outside the boxes, a residual `3 -> 1 -> 3` convolutional bottleneck
suppresses clutter-induced high-frequency responses. The refined residual is
averaged across RGB channels and used for difference-guided spatial modulation.
A shared-MLP channel-attention branch selects tiny-object-sensitive channels.

TDEM is removed in evaluation mode, so it introduces no detector-side inference
cost.

### UAV Camera Motion Compensation (UCMC)

At tracking time, sparse optical flow and robust affine estimation compensate
camera-induced displacement before confidence-stratified association. The
tracker first matches high-confidence detections and then uses low-confidence
detections to recover unmatched trajectories.

## Installation

```bash
conda create -n damot python=3.8 -y
conda activate damot
pip install -r requirements.txt
pip install cython cython_bbox
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python setup.py develop
pip install 'protobuf<3.21' 'numpy<1.24'
```

## Data layout

The AutoDL scripts use `/root/autodl-tmp` by default:

```text
/root/autodl-tmp/
|-- DAMOT-main/
`-- data/
    |-- VisDrone2019-MOT-train/
    |-- VisDrone2019-MOT-val/
    `-- VisDrone2019-MOT-test-dev/
```

Set `AUTODL_ROOT` to use a different data root.

## Pretrained weights

Place the following files in `pretrained/`:

```text
pretrained/yolox_x.pth
pretrained/veriwild_bot_R50-ibn.pth
```

- YOLOX-X initializes the detector.
- VERI-Wild BoT R50-ibn supplies ReID descriptors during tracking.

## Training

The default experiment uses five official VisDrone2019-MOT categories,
1088x1920 inputs, 20 epochs, mixed precision, and checkpoints every four
epochs.

```bash
cd /root/autodl-tmp/DAMOT-main
BATCH_SIZE=4 DEVICES=1 bash scripts/train_damot.sh
```

Checkpoints are written to `model/`:

```text
model/epoch_4_ckpt.pth.tar
model/epoch_8_ckpt.pth.tar
model/epoch_12_ckpt.pth.tar
model/epoch_16_ckpt.pth.tar
model/epoch_20_ckpt.pth.tar
model/latest_ckpt.pth.tar
model/best_ckpt.pth.tar
```

Resume from the latest periodic checkpoint:

```bash
RESUME=1 BATCH_SIZE=4 DEVICES=1 bash scripts/train_damot.sh
```

## Evaluation

VisDrone2019-MOT validation:

```bash
CHECKPOINT=/root/autodl-tmp/DAMOT-main/model/epoch_20_ckpt.pth.tar \
bash scripts/evaluate_damot.sh val
```

Test-dev:

```bash
CHECKPOINT=/root/autodl-tmp/DAMOT-main/model/epoch_20_ckpt.pth.tar \
bash scripts/evaluate_damot.sh test-dev
```

The evaluator reports `MOTA`, `IDF1`, `IDs`, `FP`, `FN`, `MT`, and `ML`, and
writes per-sequence tracking files plus CSV and JSON summaries.

## Demo

```bash
python tools/demo_track.py damot \
  -f exps/example/mot/damot_visdrone.py \
  -c model/epoch_20_ckpt.pth.tar \
  --config-file fast_reid/configs/VisDrone/damot_veriwild_R50-ibn.yml \
  --path demo/video.mp4 \
  --fp16 --fuse --save_result
```

## Main project files

```text
yolox/models/tdem.py                         TDEM and BRM
yolox/tracker/damot_tracker.py               DAMOT trajectory management
exps/example/mot/damot_visdrone.py           VisDrone training configuration
tools/eval_damot.py                          tracking evaluation entry point
scripts/train_damot.sh                       AutoDL training launcher
scripts/evaluate_damot.sh                    AutoDL evaluation launcher
```

## Citation

```bibtex
@article{damot,
  title={DAMOT: Difference-Aware Tiny-Object Enhancement for UAV Multi-Object Tracking},
  author={Author Name(s)},
  year={2026}
}
```

Replace the author and publication fields with the final paper metadata before
release.

## Acknowledgements

This implementation builds on YOLOX, ByteTrack, BoT-SORT, FastReID, SR-TOD,
and SET. We thank their authors and open-source communities.
