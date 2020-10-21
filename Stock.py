# This is a class file that stores info and methods of holding stocks
# coding: utf-8

# Import standard packages
import datetime as dt


class Stock:

    def __init__(self, stock_code, purchase_day, purchase_price, lot):
        self.stock_code = stock_code
        self.purchase_day = purchase_day  # purchase_day is the index of the purchase date in the imported data
        self.purchase_price = float(purchase_price)
        self.lot = lot

        self.current_day = purchase_day
        self.current_price = self.purchase_price
        self.can_sell = False

    def update_stock_info(self, stock_price_slice, actual_day, date_list, min_holding_day):
        # Update trading date and current price
        self.current_day = actual_day
        self.current_price = stock_price_slice[self.stock_code]

        # Update selling status
        if not self.can_sell:
            purchase_date = dt.datetime.strptime(date_list[self.purchase_day], "%Y-%m-%d")
            current_date = dt.datetime.strptime(date_list[actual_day], "%Y-%m-%d")
            self.can_sell = (current_date - purchase_date).days == min_holding_day

    def close_stock_position(self, account):
        account.cash = account.cash + self.current_price * self.lot * 100

    def get_name(self):
        return self.stock_code

    def get_stock_value(self):
        name = self.stock_code
        mv = float(self.lot * self.current_price * 100)
        print("Stock looked up: %s ------ Current Value: %f" % (name, mv))
        return mv
