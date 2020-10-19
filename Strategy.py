# This is the interface where methods considering the core strategy, such as stock/weight calculation, stock selection,
# and beta calculation are stored.
# coding: utf-8

# Import standard packages
import tensorflow as tf
import pandas as pd
import numpy as np
import math

# Import classes and interfaces
import Factor as fct
import main

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
            day_slice.append(math.log(price_data_frame.iloc[price_row, target] / price_data_frame.iloc[day, target]))
        return_data_frame.loc[day] = day_slice
    return return_data_frame


def get_rmw_factor_list(roe_list, stock_return_frame, market_capital_frame):
    # Initialize factor list
    rmw_factor_list = pd.DataFrame(np.zeros((len(roe_list.index), 1)), columns=["RMW"])

    # Calculate RMW factor
    for i in range(len(roe_list)):
        rmw_factor_list.iloc[i] = fct.RMW(roe_list.iloc[i], stock_return_frame.iloc[i], market_capital_frame.iloc[i])
    return rmw_factor_list


def get_selected_stock(stock_return_frame, factor_list):
    stock_number = len(stock_return_frame)
    index = [_ for _ in range(stock_number)]

    # Get alpha list
    alpha = __get_alpha(stock_return_frame, factor_list)
    for i in range(stock_number):
        for j in range(stock_number - i - 1):
            if alpha[j] > alpha[j + 1]:
                alpha[j], alpha[j + 1], index[j], index[j + 1] = alpha[j + 1], alpha[j], index[j + 1], index[j]

    # Select stocks to invest. Return a list of stock code strings
    selected_stock = []
    for _ in range(main.MAX_HOLDING_STOCK):
        if alpha[_] < 0:
            selected_stock.append(main.STOCK_NAME[_])
    return selected_stock


def __get_alpha(stock_return_frame, factor_list):
    stock_number = len(stock_return_frame)
    alpha_list = []

    for _ in range(stock_number):
        beta = __get_beta(factor_list, stock_return_frame.iloc[:, _])
        alpha_list[_] = stock_return_frame.iloc[-1, _] - beta * factor_list.iloc[-1]
    return alpha_list


def __get_beta(x, y):
    # Form up training data panel
    x = np.array(x).reshape(BATCH_SIZE, 1)
    y = np.array(y).reshape(BATCH_SIZE, 1)
    train_data = tf.data.Dataset.from_tensor_slices((x, y)).repeat().batch(BATCH_SIZE)

    # Start training iteration and return trained parameter
    iterator = train_data.make_one_shot_iterator()
    return __backward(iterator)


def __backward(iterator):
    # Define forward graph
    x = tf.placeholder(tf.float32, shape=(None, INPUT_NODE))
    y = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))
    beta = tf.Variable(tf.ones([1], dtype=tf.float32))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(REGULARIZER)(beta))
    y_estimate = beta * x

    # Define loss function
    with tf.name_scope("loss"):
        mse = tf.reduce_mean(tf.square(y - y_estimate))
        lossMSE = tf.reduce_mean(mse)
        loss = lossMSE + tf.add_n(tf.get_collection("losses"))

    # Define training process
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        1,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    with tf.name_scope("EMA"):
        ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_step)
        ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="trainBeta")

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEP):
            x, y = sess.run(iterator.get_next())
            _, lossValue, step = sess.run([train_op, loss, global_step], feed_dict={x: x, y: y})

        if i % 50 == 0:
            print("Step %d: Loss---%f" % (i, lossValue))
    return sess.run(beta)


def get_stock_weight(leverage):
    # The initial weight allocated to stocks under value hedging rule.
    return 1 / (1 + leverage)  # stock weight, future weight


def get_future_weight(leverage, holding_stock, stock_return_frame, index_return_frame):
    # Calculated weighted average return of the stock portfolio
    holding_value = sum(_.get_stock_value() for _ in holding_stock)
    stock_portfolio_return = sum(stock_return_frame[_.get_name()] * _.get_stock_value() / holding_value
                                 for _ in holding_stock)

    # Calculate the market beta
    beta = __get_beta(index_return_frame, stock_portfolio_return)

    # Return future weight under beta hedging rule
    future_weight = beta * leverage / (beta * leverage + 1)
    return future_weight
