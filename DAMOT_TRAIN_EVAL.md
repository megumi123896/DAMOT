# DAMOT VisDrone2019-MOT training and evaluation

## Model path

Training uses:

```text
image -> YOLOX-X backbone/PAFPN -> (P3, P4, P5)
      -> TDEM
         |-- BRM reconstruction
         |-- protected difference-domain background smoothing
         `-- difference-guided spatial-channel enhancement
      -> (enhanced P3, P4, P5)
      -> YOLOX head and detection loss
      + RGB reconstruction MSE
```

Evaluation uses:

```text
image -> YOLOX-X backbone/PAFPN -> (P3, P4, P5)
      -> YOLOX head -> detections
      -> UCMC -> confidence-stratified association -> tracks
```

TDEM is not executed in `model.eval()` mode.

## 1. Prepare weights

Expected files:

```text
pretrained/yolox_x.pth
pretrained/veriwild_bot_R50-ibn.pth
```

## 2. Train DAMOT

Linux/AutoDL:

```bash
cd /root/autodl-tmp/DAMOT-main
BATCH_SIZE=4 DEVICES=1 bash scripts/train_damot.sh
```

Python entry point after annotation conversion:

```bash
python tools/train.py \
  -f exps/example/mot/damot_visdrone.py \
  -d 1 -b 4 --fp16 \
  -c pretrained/yolox_x.pth
```

Resume:

```bash
RESUME=1 BATCH_SIZE=4 DEVICES=1 bash scripts/train_damot.sh
```

## 3. Evaluate DAMOT

Validation:

```bash
CHECKPOINT=/root/autodl-tmp/DAMOT-main/model/epoch_20_ckpt.pth.tar \
bash scripts/evaluate_damot.sh val
```

Test-dev:

```bash
CHECKPOINT=/root/autodl-tmp/DAMOT-main/model/epoch_20_ckpt.pth.tar \
bash scripts/evaluate_damot.sh test-dev
```

The evaluator reports `MOTA`, `IDF1`, `IDs`, `FP`, `FN`, `MT`, and `ML`.
Tracker predictions covered by ignored regions by at least 50 percent are
removed before accumulation. Ground truth is restricted to pedestrian, car,
van, truck, and bus.
