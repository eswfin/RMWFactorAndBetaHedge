# This is the main file of the project where I/O, main function, and all the parameters are contained.

import pandas as pd
import datetime


# -------------------- Parameter Setting -------------------- #
# 0. Time parameters
START_DATE = '2010-04-30'
END_DATE = '2020-06-30'

# 1. Account parameters
INITIAL_CAPITAL = 10000000

# 2. Trading parameters
LEVERAGE = 0.12

# 3. Strategy parameters
MAX_HOLDING_STOCK = 3
MIN_HOLDING_DAY = 365
CALC_WINDOW = 36  # The time period used in factor and beta calculations

# 4. Path parameters
DATA_STORE_PATH = './output/'


# --------------------------- I/O ---------------------------- #
STOCK_PRICE = pd.read_csv('componentPrice', index_col='Date')
FUTURE_PRICE = pd.read_csv('IF.csv', index_col='Date')
INDEX_PRICE = pd.read_csv('HS300.csv', index_col='Date')
MKT_CAP = pd.read_csv('marketValue.csv', index_col='Date')
ROE = pd.read_csv('ROE.csv', index_col='Date')


# ----------------- Other Parameters Inferred ----------------- #
DAYS = len(FUTURE_PRICE) - CALC_WINDOW - 1  # Actual trading days in back-test considering data loss
STOCK_NAME = list(STOCK_PRICE.index)


def main():

    return -1


if __name__ == 'main':
    main()
