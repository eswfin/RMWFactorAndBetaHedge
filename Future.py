# This is a class file that stores info and methods of holding futures
# coding: utf-8

class Future:

    def __init__(self, purchase_price, lot):
        self.purchase_price = purchase_price
        self.lot = lot
        self.current_price = purchase_price

    def update_future_info(self, future_price):
        self.current_price = future_price

    def close_future_position(self, account):
        account.cash = account.cash + (self.purchase_price - self.current_price) * self.lot * 100
