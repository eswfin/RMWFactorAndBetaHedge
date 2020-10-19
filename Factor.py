# This is the interface where stores factor calculation methods
# coding: utf-8

import math


def RMW(roe_slice, stock_return_slice, market_capital_slice):
    # Reformat data
    roe = list(roe_slice)
    stock_return = list(stock_return_slice)
    mkt_cap = list(market_capital_slice)

    # Sorting
    stock_number = len(roe)
    index = [i for i in range(stock_number)]
    for i in range(stock_number):
        for j in range(stock_number - i - 1):
            if roe[j] > roe[j + 1]:
                roe[j], roe[j + 1] = roe[j + 1], roe[j]
                index[j], index[j + 1] = index[j + 1], index[j]

    # Calculate market value sums of each group
    group_size = math.floor(stock_number / 3)
    weak_mkt_cap = sum(mkt_cap[col_index] for col_index in index[0: group_size])
    robust_mkt_cap = sum(mkt_cap[col_index] for col_index in index[2 * group_size: stock_number])

    # Calculate weighted average return of each group
    weak_return = sum(stock_return[col_index] * mkt_cap[col_index] / weak_mkt_cap for col_index in index[0: group_size])
    robust_return = sum(stock_return[col_index] * mkt_cap[col_index] / robust_mkt_cap
                        for col_index in index[2 * group_size: stock_number])

    # Return RMW factor of the specific day
    return robust_return - weak_return
