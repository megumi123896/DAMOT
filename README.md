<h1 align="center">DAMOT</h1>

<h3 align="center">Difference-Aware Tiny-Object Enhancement<br>for UAV Multi-Object Tracking</h3>

<p align="center">
  Zhengkun Zhang<sup>1</sup> &nbsp;·&nbsp;
  Ying Zhao<sup>1,*</sup> &nbsp;·&nbsp;
  Ming Li<sup>2,*</sup> &nbsp;·&nbsp;
  Zhimin Chen<sup>1</sup> &nbsp;·&nbsp;
  Rong Ye<sup>1</sup> &nbsp;·&nbsp;
  Shuo Zhang<sup>2,3,*</sup>
</p>

<p align="center">
  <sup>1</sup> School of Aerospace Information Science and Technology, Shanghai Dianji University<br>
  <sup>2</sup> School of Artificial Intelligence and Data Science, Shanghai Dianji University<br>
  <sup>3</sup> School of Computer Science, Shanghai Jiao Tong University<br>
  <sub>* Corresponding authors</sub>
</p>

<p align="center">
  <a href="#overview"><img src="https://img.shields.io/badge/Task-UAV_Multi--Object_Tracking-2563eb?style=flat-square" alt="UAV multi-object tracking"></a>
  <a href="#results"><img src="https://img.shields.io/badge/Benchmark-VisDrone2019-0f766e?style=flat-square" alt="VisDrone2019 benchmark"></a>
  <a href="#results"><img src="https://img.shields.io/badge/Benchmark-UAVDT-0f766e?style=flat-square" alt="UAVDT benchmark"></a>
  <a href="#getting-started"><img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=flat-square" alt="PyTorch implementation"></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp; / &nbsp;
  <a href="#framework">Framework</a> &nbsp; / &nbsp;
  <a href="#qualitative-comparison">Visual Comparisons</a> &nbsp; / &nbsp;
  <a href="#results">Results</a> &nbsp; / &nbsp;
  <a href="#getting-started">Getting Started</a> &nbsp; / &nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center"><b>Stronger tiny-object observations. More consistent trajectories in moving UAV views.</b></p>

<table align="center">
  <tr>
    <th align="center">VisDrone2019 test-dev</th>
    <th align="center">UAVDT</th>
    <th align="center">Training-only enhancement</th>
  </tr>
  <tr>
    <td align="center"><b>51.94 MOTA · 65.24 IDF1</b><br><sub>+4.74 / +3.14 points vs. SFTrack</sub></td>
    <td align="center"><b>55.7 MOTA · 73.1 IDF1</b><br><sub>+0.9 / +0.9 points vs. DMTrack</sub></td>
    <td align="center"><b>No added detector inference cost</b><br><sub>TDEM is bypassed at inference</sub></td>
  </tr>
</table>

## Overview

UAV videos contain dense tiny objects, weak appearance cues, frequent occlusions, and large camera-induced displacements. **DAMOT** addresses these challenges with a YOLOX-X detector and UAV-aware trajectory association.

During training, a **Training-time Difference Enhancement Module (TDEM)** reconstructs the input image from high-resolution features and uses the reconstruction residual to recover fine-grained target cues. Target-preserving background smoothing and spatial-channel attention strengthen tiny-object representations. During tracking, **UAV Camera Motion Compensation (UCMC)** corrects camera-induced displacement before confidence-stratified matching.

- **Recover tiny-object detail.** Bilinear reconstruction exposes information lost during feature extraction.
- **Suppress background interference.** Refined residuals guide spatial modulation and channel attention while preserving target regions.
- **Maintain trajectory continuity.** Camera-compensated prediction and two-stage association recover tracks from weak detections.

## Framework

<p align="center">
  <a href="assets/damot_overview.png">
    <img src="assets/damot_overview.png" width="100%" alt="DAMOT architecture: training-time difference enhancement with BRM, background smoothing and spatial-channel attention, followed by UCMC and confidence-stratified association.">
  </a>
</p>

<p align="center"><sub><b>DAMOT architecture.</b> TDEM enhances the highest-resolution feature during training; UCMC and confidence-stratified association preserve trajectories during tracking.</sub><br><sub><a href="assets/damot_overview.png">High-resolution PNG</a> · <a href="assets/damot_overview.pdf">Original PDF</a></sub></p>

| Component | Role | Active stage |
| :--- | :--- | :--- |
| **BRM** | Reconstructs RGB from stride-8 `P3` using three bilinear upsampling blocks | Training |
| **TDEM** | Refines reconstruction residuals and applies spatial-channel enhancement to `P3` | Training |
| **UCMC** | Uses sparse optical flow and robust affine estimation to compensate camera motion | Tracking |
| **Two-stage association** | Matches high-confidence detections first, then uses low-confidence detections to recover unmatched tracks | Tracking |

<details>
<summary><b>How the training and inference paths differ</b></summary>

```text
Training
Image -> YOLOX-X backbone + PAFPN -> (P3, P4, P5)
                                      |
                               TDEM enhances P3
                                      |
                              YOLOX detection head
Loss = detection loss + RGB reconstruction loss

Inference
Image -> YOLOX-X backbone + PAFPN -> YOLOX detection head
      -> UCMC -> confidence-stratified association -> tracks
```

TDEM preserves reconstruction residuals inside ground-truth boxes and smooths background residuals through a `3 -> 1 -> 3` convolutional bottleneck. The refined residual guides spatial modulation, while a shared-MLP attention branch selects discriminative channels. `P4` and `P5` retain their original features.

In `model.eval()` mode, the detector bypasses TDEM and feeds the original multi-scale features directly into the detection head. No annotations or reconstruction branch are needed for inference. The claim of no added inference cost refers specifically to the detector-side TDEM branch.

</details>

## Qualitative Comparison

<p align="center">
  <a href="assets/qualitative_comparison.png">
    <img src="assets/qualitative_comparison.png" width="100%" alt="AMF-MOT, STDFormer, SFTrack and DAMOT compared over three consecutive frames in small-object scenes and scenes with viewpoint variation.">
  </a>
</p>

<p align="center"><sub><b>Tracking across consecutive frames.</b> Comparison with AMF-MOT, STDFormer, and SFTrack under tiny-object and viewpoint-change challenges.</sub><br><sub><a href="assets/qualitative_comparison.png">High-resolution PNG</a> · <a href="assets/qualitative_comparison.pdf">Original PDF</a></sub></p>

| Small-object scenes | Viewpoint variation |
| :--- | :--- |
| DAMOT maintains tiny-pedestrian observations and consistent identities across adjacent frames. | DAMOT preserves the vehicle's **ID 49** under camera-induced displacement, where competing methods lose the match and initialize new identities. |

## Results

The following numbers are **reported in the DAMOT manuscript**. Higher is better for MOTA, IDF1, and MT; lower is better for IDs, FP, FN, and ML. Bold indicates the best value among the methods listed in each table.

### VisDrone2019 test-dev

| Method | MOTA ↑ | IDF1 ↑ | IDs ↓ | FP ↓ | FN ↓ | MT ↑ | ML ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UAVMOT | 36.1 | 51.0 | 2,775 | 27,983 | 115,925 | 520 | 574 |
| STCMOT | 41.2 | 52.0 | 3,984 | 36,428 | 94,445 | 667 | 453 |
| AMF-MOT | 41.2 | 52.8 | 3,617 | 37,632 | 93,820 | 673 | 257 |
| ByteTrack | 42.3 | 53.4 | 989 | 15,689 | 117,539 | 586 | 675 |
| DroneMOT | 43.7 | 58.6 | 1,112 | 41,998 | **86,177** | 689 | 397 |
| STDFormer | 45.9 | 57.1 | 1,440 | 21,288 | 101,506 | 684 | 538 |
| SFTrack | 47.2 | 62.1 | **557** | 27,159 | 94,910 | 753 | 518 |
| **DAMOT (ours)** | **51.94** | **65.24** | 1,114 | **15,023** | 94,189 | **782** | **211** |

### UAVDT

| Method | MOTA ↑ | IDF1 ↑ |
| :--- | ---: | ---: |
| DeepSORT | 40.7 | 58.2 |
| UAVMOT | 46.4 | 67.3 |
| STCMOT | 49.2 | 69.8 |
| DroneMOT | 50.1 | 69.6 |
| DMTrack | 54.8 | 72.2 |
| HA-Tracker | 52.2 | 72.3 |
| EGEN | 46.6 | 60.1 |
| **DAMOT (ours)** | **55.7** | **73.1** |

<details>
<summary><b>Tiny-object detection and component ablations</b></summary>

**Scale-specific detection AP (%) on VisDrone2019.** All models in this comparison are trained on VisDrone2019-DET. Tiny, very-tiny, and small correspond to 8-16, 2-8, and 16-32 pixels, respectively.

| Method | AP (tiny) ↑ | AP (very-tiny) ↑ | AP (small) ↑ |
| :--- | ---: | ---: | ---: |
| FCOS | 4.5 | 0.7 | 15.7 |
| YOLOX | 10.6 | 4.3 | 21.2 |
| RFLA | 13.0 | 4.5 | 23.6 |
| **YOLOX + TDEM** | **14.2** | **5.1** | **27.1** |

**Component ablation on VisDrone2019.** The two modules address complementary sources of tracking error.

| TDEM | UCMC | IDF1 ↑ | MOTA ↑ | IDs ↓ |
| :---: | :---: | ---: | ---: | ---: |
| - | - | 54.61 | 45.82 | 4,057 |
| ✓ | - | 60.32 | 50.02 | 1,293 |
| - | ✓ | 57.56 | 48.28 | 1,803 |
| ✓ | ✓ | **65.24** | **51.94** | **1,114** |

</details>

## Getting Started

The commands below cover the included **VisDrone2019-MOT** training and evaluation workflow. See [DAMOT_TRAIN_EVAL.md](DAMOT_TRAIN_EVAL.md) for additional implementation notes. Model weights and datasets are not bundled with the source code.

### 1. Installation

Use a Linux environment with Conda and a CUDA-compatible PyTorch installation. The repository's existing setup uses Python 3.8; select a matching PyTorch/torchvision build for your GPU and CUDA environment before installing the remaining dependencies.

```bash
git clone https://github.com/megumi123896/DAMOT.git
cd DAMOT

conda create -n damot python=3.8 -y
conda activate damot

# Install a compatible PyTorch + torchvision build first.
pip install -r requirements.txt
pip install cython cython_bbox
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
python setup.py develop
pip install 'protobuf<3.21' 'numpy<1.24'
```

### 2. Prepare data and pretrained weights

Download and extract the VisDrone2019-MOT splits. The launchers default to the following data layout:

```text
/root/autodl-tmp/
└── data/
    ├── VisDrone2019-MOT-train/
    ├── VisDrone2019-MOT-val/
    └── VisDrone2019-MOT-test-dev/
```

Set `AUTODL_ROOT` to change the parent of `data/`, or set `TRAIN_ROOT`, `VAL_ROOT`, and `TEST_DEV_ROOT` individually. The launchers call [`tools/prepare_visdrone.py`](tools/prepare_visdrone.py) to prepare COCO-format annotations under `datasets/visdrone/`.

Place the initialization weights under the repository root:

```text
pretrained/
├── yolox_x.pth                # COCO-pretrained YOLOX-X detector
└── veriwild_bot_R50-ibn.pth    # FastReID appearance descriptors
```

`DETECTOR_PRETRAINED` overrides the detector initialization path for training. `REID_WEIGHTS` overrides the ReID weight path for evaluation.

### 3. Train

Run from the repository root:

```bash
BATCH_SIZE=4 DEVICES=1 bash scripts/train_damot.sh
```

The DAMOT experiment uses **5 VisDrone MOT categories**, a configured input size of **1088 × 1920**, **20 epochs**, mixed precision, and a checkpoint interval of **4 epochs**. The example above sets batch size 4; the launcher defaults to 1 if `BATCH_SIZE` is omitted.

<details>
<summary><b>Checkpoints, resuming, and a custom data root</b></summary>

```text
model/
├── epoch_4_ckpt.pth.tar
├── epoch_8_ckpt.pth.tar
├── epoch_12_ckpt.pth.tar
├── epoch_16_ckpt.pth.tar
├── epoch_20_ckpt.pth.tar
├── latest_ckpt.pth.tar
└── best_ckpt.pth.tar
```

Resume training:

```bash
RESUME=1 BATCH_SIZE=4 DEVICES=1 bash scripts/train_damot.sh
```

Use datasets located under `/path/to/storage/data/`:

```bash
AUTODL_ROOT=/path/to/storage BATCH_SIZE=4 DEVICES=1 \
bash scripts/train_damot.sh
```

</details>

### 4. Evaluate

```bash
# VisDrone2019-MOT validation
CHECKPOINT="$PWD/model/epoch_20_ckpt.pth.tar" \
bash scripts/evaluate_damot.sh val

# VisDrone2019-MOT test-dev
CHECKPOINT="$PWD/model/epoch_20_ckpt.pth.tar" \
bash scripts/evaluate_damot.sh test-dev
```

The evaluator reports **MOTA, IDF1, IDs, FP, FN, MT, and ML**, and writes per-sequence tracking files together with CSV and JSON summaries. When `CHECKPOINT` is omitted, the launcher uses `model/best_ckpt.pth.tar`.

### 5. Track a video

Provide a video at `demo/video.mp4`, or replace `--path` with your own video path:

```bash
python tools/demo_track.py damot \
  -f exps/example/mot/damot_visdrone.py \
  -c model/epoch_20_ckpt.pth.tar \
  --config-file fast_reid/configs/VisDrone/damot_veriwild_R50-ibn.yml \
  --path demo/video.mp4 \
  --fp16 --fuse --save_result
```

## Repository Guide

| File | Description |
| :--- | :--- |
| [`yolox/models/tdem.py`](yolox/models/tdem.py) | TDEM, BRM, background smoothing, and spatial-channel enhancement |
| [`yolox/tracker/ucmc.py`](yolox/tracker/ucmc.py) | UAV camera-motion compensation |
| [`yolox/tracker/damot_tracker.py`](yolox/tracker/damot_tracker.py) | Trajectory management and data association |
| [`exps/example/mot/damot_visdrone.py`](exps/example/mot/damot_visdrone.py) | DAMOT detector and VisDrone experiment configuration |
| [`tools/prepare_visdrone.py`](tools/prepare_visdrone.py) | Dataset and annotation preparation |
| [`tools/eval_damot.py`](tools/eval_damot.py) | Tracking evaluation entry point |
| [`scripts/train_damot.sh`](scripts/train_damot.sh) | Training launcher |
| [`scripts/evaluate_damot.sh`](scripts/evaluate_damot.sh) | Evaluation launcher |
| [`assets/`](assets/) | Framework and qualitative-comparison figures in PNG and PDF |

## Citation

If you find DAMOT useful, please cite the manuscript. Publication details can be added when available.

```bibtex
@misc{damot,
  title  = {DAMOT: Difference-Aware Tiny-Object Enhancement for UAV Multi-Object Tracking},
  author = {Zhang, Zhengkun and Zhao, Ying and Li, Ming and Chen, Zhimin and Ye, Rong and Zhang, Shuo},
  note   = {Manuscript},
  url    = {https://github.com/megumi123896/DAMOT}
}
```

## Acknowledgements

This implementation builds on YOLOX, ByteTrack, BoT-SORT, FastReID, SR-TOD, and SET. We thank their authors and the open-source community for making their work available.

<p align="right"><a href="#damot">Back to top ↑</a></p>
