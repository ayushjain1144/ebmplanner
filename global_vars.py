"""Define global variables here."""

import numpy as np

from cameras import RealSenseD415


PIXEL_SIZE = 0.003125 / 2
CAMERA_CONFIG = RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [-0.1, 0.28]]).astype(np.float32)

IN_SHAPE = (640, 320, 6)
