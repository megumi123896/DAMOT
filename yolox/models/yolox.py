#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, tdem=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head
        self.tdem = tdem

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            reconstruction_loss = None
            if self.tdem is not None:
                enhanced_p3, reconstruction_loss = self.tdem(
                    fpn_outs[0], x, targets
                )
                fpn_outs = (enhanced_p3, fpn_outs[1], fpn_outs[2])

            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            if reconstruction_loss is not None:
                loss = loss + reconstruction_loss
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            if reconstruction_loss is not None:
                outputs["reconstruction_loss"] = reconstruction_loss
        else:
            # DAMOT removes TDEM from the inference path.
            outputs = self.head(fpn_outs)

        return outputs
