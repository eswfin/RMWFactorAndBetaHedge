# This is a class file that stores info and methods of holding stocks

class Stock:

    def __init__(self, stock_code, purchase_day, purchase_price, lot):
        self.stock_code = stock_code
        self.purchase_day = purchase_day  # purchase_day is the index of the purchase date in the imported data
        self.purchase_price = purchase_price
        self.lot = lot

        self.current_day = purchase_day
        self.current_price = purchase_price
        self.can_sell = False
