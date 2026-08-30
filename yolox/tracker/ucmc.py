#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""UAV Camera Motion Compensation (UCMC) used by DAMOT."""

from .gmc import GMC


class UAVCameraMotionCompensation(GMC):
    """Estimate inter-frame camera motion from sparse optical flow."""

    pass
