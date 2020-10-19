# This is the interface where methods considering the core strategy, such as stock/weight calculation, stock selection,
# and beta calculation are stored

import tensorflow as tf
import numpy as np


# ----------------- Hyper-parameter Setting ----------------- #
STEP = 200
SEED = 100
BATCH_SIZE = 36  # To avoid cross reference, manually set as equal to CALC_WINDOW in main.py
INPUT_NODE = 1
OUTPUT_NODE = 1
REGULARIZER = 0.001
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.95
MOVING_AVERAGE_DECAY = 0.95


def get_initial_weight():

    return -1
