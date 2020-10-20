# This is a class file that stores info and methods of the investment account, including cash, stocks, and futures
# coding: utf-8

# Import customized classes
import Stock as stk
import Future as ft


class Account:

    def __init__(self, cash):
        self.cash = cash
        self.stock = []
        self.future = []

    def update_position_info(self, stock_price_frame, future_price_list, actual_day, date_list, min_holding_day):
        # Update stock info
        holding_stock_number = len(self.stock)
        if holding_stock_number != 0:
            for _ in self.stock:
                _.update_stock_info(stock_price_frame.iloc[-1], actual_day, date_list, min_holding_day)

        # Update future info
        if len(self.future):
            self.future[0].update_future_info(future_price_list.iloc[-1])

    def get_market_value(self):
        # Note that in main function, at the place where this function is called, the account only has cash and stocks
        return self.cash + sum(_.get_stock_value() for _ in self.stock)

    def buy_stock(self, stock_code, purchase_day, purchase_price, lot):
        new_stock = stk.Stock(stock_code, purchase_day, purchase_price, lot)
        self.stock.append(new_stock)
        self.cash = self.cash - new_stock.get_stock_value()

    def short_future(self, price, lot, leverage):
        self.future.append(ft.Future(price, lot))
        self.cash = self.cash - price * lot * 100 * leverage
