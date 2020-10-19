# This is a class file that stores info and methods of the investment account, including cash, stocks, and futures

class Account:

    def __init__(self, cash):
        self.cash = cash
        self.stock = []
        self.future = []
