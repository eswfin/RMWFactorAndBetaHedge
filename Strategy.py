# This is the interface where methods considering the core strategy, such as stock/weight calculation, stock selection,
# and beta calculation are stored

import tensorflow as tf
import pandas as pd
import numpy as np
import math


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


# The method for generating return data from price data frame
def get_return_data(price_data_frame):
    target_name = list(price_data_frame.columns)
    target_number = len(target_name)
    price_data_len = len(price_data_frame)

    # Start calculation. The return rate is calculated using logarithm return method.
    return_data_len = price_data_len - 1
    return_data_frame = pd.DataFrame(np.zeros((return_data_len, target_number)), columns=target_name)  # Init frame
    for day in range(return_data_len):
        price_row = day + 1
        day_slice = []
        for target in range(target_number):
            day_slice[target] = math.log(price_data_frame.iloc[price_row, target] / price_data_frame.iloc[day, target])
        return_data_frame.loc[day] = day_slice
    return return_data_frame


def get_initial_weight():

    return -1
