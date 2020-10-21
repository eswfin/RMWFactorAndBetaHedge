# This is the main file of the project where I/O, main function, and all the parameters are contained.
# coding: utf-8

# Import standard packages
import math
import time
import numpy as np
import pandas as pd

# Import customised classes and interfaces
import Account as acc
import Strategy as strt

# -------------------- Parameter Setting -------------------- #
# 0. Time parameters
START_DATE = '2010-04-30'
END_DATE = '2020-06-30'

# 1. Account parameters
INITIAL_CAPITAL = 10000000

# 2. Trading parameters
LEVERAGE = 0.12

# 3. Strategy parameters
MAX_HOLDING_STOCK = 10
MIN_HOLDING_DAY = 365
CALC_WINDOW = 36  # The time period used in factor and beta calculations

# 4. Path parameters
INPUT_PATH = './Input/'
OUTPUT_PATH = './Output/'


# --------------------------- I/O ---------------------------- #
STOCK_PRICE = pd.read_csv(INPUT_PATH + 'componentPrice.csv', index_col='Date')
FUTURE_PRICE = pd.read_csv(INPUT_PATH + 'IF.csv', index_col='Date')
INDEX_PRICE = pd.read_csv(INPUT_PATH + 'HS300.csv', index_col='Date')
MKT_CAP = pd.read_csv(INPUT_PATH + 'marketValue.csv', index_col='Date')
ROE = pd.read_csv(INPUT_PATH + 'ROE.csv', index_col='Date')


# ----------------- Other Parameters Inferred ----------------- #
DAYS = len(FUTURE_PRICE) - CALC_WINDOW - 1  # Actual trading days in back-test considering data loss
DATE_LIST = list(STOCK_PRICE.index)[1:]  # For back-test use
STOCK_NAME = list(STOCK_PRICE.columns)


# ---------------- Adjusted Data for Back-test ---------------- #
stock_price, future_price, index_price = STOCK_PRICE.iloc[1:, :], FUTURE_PRICE.iloc[1:], INDEX_PRICE.iloc[1:]
stock_return, index_return = strt.get_return_data(STOCK_PRICE), strt.get_return_data(INDEX_PRICE)
roe, mkt_cap = ROE.iloc[1:, :], MKT_CAP.iloc[1:, :]
rmw = strt.get_rmw_factor_list(roe, stock_return, mkt_cap)


def main():
    user_interface()

    # Set global time variables
    trading_day = 0

    # Initialize performance data frame
    performance = []

    # Start back-test
    account = acc.Account(INITIAL_CAPITAL)
    while trading_day <= DAYS:
        performance.append(daily_execution(trading_day, account))
        trading_day = trading_day + 1

    # Calculate net values and save back-test result to files
    print('****** Back-test finished, generating output data ******')
    columns = ['Account', 'HS300']
    performance = pd.DataFrame(performance, columns=columns)
    performance_return = strt.get_return_data(performance)
    net_value = pd.DataFrame(np.ones((len(performance_return), 2)), columns=columns)
    net_value.iloc[0] = [1, 1]
    for i in range(1, len(net_value)):
        net_value.iloc[i] = net_value.iloc[i - 1] * (net_value.iloc[0] + performance_return.iloc[i - 1])
    performance.to_csv(OUTPUT_PATH + 'performance.csv')
    performance_return.to_csv(OUTPUT_PATH + 'return.csv')
    net_value.to_csv(OUTPUT_PATH + 'netValue.csv')

    # Print performance data
    performance_management(performance_return, net_value)


def daily_execution(trading_day, account):
    actual_day = trading_day + CALC_WINDOW - 1  # Actual index corresponds to the price data frames
    print('******* On %s -- Day %d *******' % (DATE_LIST[actual_day], trading_day))

    # Generate required moving data for factor and beta calculation
    stock_price_moving = stock_price.iloc[actual_day - CALC_WINDOW + 1: actual_day + 1]
    future_price_moving = future_price.iloc[actual_day - CALC_WINDOW + 1: actual_day + 1]
    stock_return_moving = stock_return.iloc[actual_day - CALC_WINDOW + 1: actual_day + 1]
    index_return_moving = index_return.iloc[actual_day - CALC_WINDOW + 1: actual_day + 1]
    rmw_moving = rmw.iloc[actual_day - CALC_WINDOW + 1: actual_day + 1]

    # Update latest price information
    account.update_position_info(stock_price_moving, future_price_moving, actual_day, DATE_LIST, MIN_HOLDING_DAY)

    # Get current holdings in position
    holding_stock = account.stock
    holding_future = account.future
    holding_list = [_.get_name() for _ in holding_stock]
    buy_flag = 0
    if trading_day != 0:
        # Check if on the day can sell any holding stock, and sell any sellable asset
        for _ in holding_stock:
            if _.can_sell:
                _.close_stock_position(account)
                holding_stock.remove(_)
                buy_flag = buy_flag + 1
        holding_future[0].close_future_position(account)
        holding_future.remove(holding_future[0])

    # Update position changes to account
    account.stock = holding_stock
    account.future = holding_future

    total_asset = account.get_market_value()  # Only cash and stocks remains in account at this stage
    print('            Account value: %.2f\n' % total_asset)

    # Only buy new stock when at least one holding stock is sold
    if buy_flag != 0 or trading_day == 0:
        # Get selected stocks to invest according to RMW factor selection result
        invest_list = strt.get_selected_stock(stock_return_moving, rmw_moving, MAX_HOLDING_STOCK)

        # Get capital allocated to stock
        stock_weight = strt.get_stock_weight(LEVERAGE)
        stock_capital = total_asset * stock_weight

        # Filter stocks in holding list from invest list
        for _ in invest_list:
            if _ in holding_list:
                invest_list.remove(_)

        # Calculate capital to invest into each new stock
        holding_stock_value = sum(_.get_stock_value() for _ in holding_stock)
        fund_per_new_stock = (stock_capital - holding_stock_value) / len(invest_list)

        # Invest new stocks
        for _ in invest_list:
            price = stock_price_moving[_][-1]
            lot = math.floor(fund_per_new_stock / price / 100)
            purchase_day = actual_day
            account.buy_stock(_, purchase_day, price, lot)

    # Invest futures using beta hedging method
    future_weight = float(strt.get_future_weight(LEVERAGE, account.stock, stock_return_moving, index_return_moving))
    theoretical_future_fund = float(total_asset * future_weight)
    price = float(future_price_moving.iloc[-1])
    cash = float(account.cash)
    lot = math.floor((min(theoretical_future_fund, cash) / LEVERAGE) / (price * 100.00))
    account.short_future(price, lot, LEVERAGE)

    return [total_asset, float(index_price.iloc[trading_day])]


def user_interface():
    print('\n\n========================================\n'
          'Welcome to the back-test analysis system\n'
          '========================================\n'
          '                    Author: Xin Yi\n'
          '\n'
          'The system will carry out a high number\n'
          'of machine learning calculations. Please\n'
          'make sure that you have carefully read\n'
          'the instructions in README.md, and set\n'
          'appropriate hyper-parameters according\n'
          'to your computer condition, in case of\n'
          'hardware damage.\n\n'
          'Please also be advised that the runtime\n'
          'can be significantly long due to the\n'
          'heavy calculation, which may also drag\n'
          'down your computer performances.\n')

    select = True
    while select:
        choice = str(input('Proceed with caution? (Y/N)\n'))
        if choice == 'y' or choice == 'Y':
            print('The program starts after 5-second countdown.\n')
            for i in range(5):
                print(5-i)
                time.sleep(1)
            select = False
            print('\n======== Back-test starts from now ========\n')
            print('-- Total trading days: %d\n' % DAYS)
            print('-- Start date: ' + START_DATE + '\n')
            print('--Initial capital: %d\n\n' % INITIAL_CAPITAL)

        elif choice == 'n' or choice == 'N':
            print('Your prudent decision is very much\n'
                  'appreciated. See you when you are ready!\n')
            select = False
            exit()
        else:
            print('That is not a valid input. Please try again:')


def performance_management(performance_return, net_value):
    sharpe = (performance_return['Account'].mean() * np.sqrt(12) / performance_return.std())['Account']
    withdraw = 0
    max_withdraw = 0
    for i in range(DAYS):
        for j in range(i, DAYS):
            if net_value['Account'][j] < net_value['Account'][i]:
                withdraw = (net_value['Account'][i] - net_value['Account'][j]) / net_value['Account'][i]
            if withdraw > max_withdraw:
                max_withdraw = withdraw
    hpr = (net_value['Account'][DAYS - 1] - net_value['Account'][0]) / net_value['Account'][0]
    print('Holding Period Return: %.2f%%' % (hpr * 100))
    print('Annual Return: %.2f%%' % (hpr * 12 * 100 / (DAYS - 1)))
    print('Annual Sharpe: %.2f' % sharpe)
    print('Maximum Withdraw: %.2f' % max_withdraw)


if __name__ == '__main__':
    main()
