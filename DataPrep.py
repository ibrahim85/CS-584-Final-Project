import numpy as np
import pandas as pd


class StockCorrelation:
    def __init__(self, closing_stock_price):
        self.stock_prices = closing_stock_price
        self.stock_covariances = [None] * np.size(closing_stock_price, axis=0)
        self.indexed_stocks = [None] * np.size(closing_stock_price, axis=0)

        self.num_stocks = np.size(self.stock_covariances, axis=0)

        self.calculate_covariance()
        self.sort()

    def calculate_covariance(self):
        self.stock_covariances = np.corrcoef(self.stock_prices)

    def sort(self):
        stock_list = np.arange(self.num_stocks)

        for index, stock in enumerate(self.stock_covariances):
            # subtract one, because we dont want to count the stock, as it is has a correlation of 1 to itself
            temp_index = [None] * (self.num_stocks -1)

            # gets the indices of the smallest to the largest stock
            positions = stock.argsort()


            # if we hit this stock, we need to adjust the remaining values and not add to our list
            # e.g. Stock A should not be included in a list of stocks most similar to Stock A
            skipped_current = 0

            for i, _ in enumerate(stock_list):
                # the list generated goes from smallest to largest, want to return a list of stocks
                # most similar to least similar
                # reverse_index will iterator through our generated list: positions[]
                reverse_index = self.num_stocks - i - 1

                # found ourselves, so skip
                if stock_list[positions[reverse_index]] == index:
                    skipped_current -= 1

                else:
                    # look up the index of next most similar stock and get the stock
                    temp_index[i + skipped_current] = stock_list[positions[reverse_index]]

            self.indexed_stocks[index] = temp_index

    def get_similar_stock_list(self, stock_index):
        if 0 <= stock_index < self.num_stocks:
            return self.indexed_stocks[stock_index]
        else:
            raise "Use a number between 0 and number of stocks-1"

class DataFromFile:

    # stock data order: close,volume,open,high,low
    def __init__(self):
        self.stock_c_v_o_h_l = [None]*3
        data = np.array(pd.read_csv('ClassTestData/APPL.csv',header=None,skiprows=2))
        self.stock_c_v_o_h_l[0] = np.array(data[:,(1,2,3)])
        self.apple_class = np.array(data[:,(5)])

        data = np.array(pd.read_csv('ClassTestData/MSFT.csv',header=None,skiprows=2))
        self.stock_c_v_o_h_l[1] = np.array(data[:,(1,2,3)])

        data = np.array(pd.read_csv('ClassTestData/XOM.csv',header=None,skiprows=2))
        self.stock_c_v_o_h_l[2] = np.array(data[:,(1,2,3)])

        self.sample_params = None
        self.sample_class = None

        self.apple_similar = None

        self.num_stocks = 3
        self.apple_index = 0
        self.msft_index = 1
        self.xom_index = 2

    def calc_stock_cor_for_apple(self):

        # get correlation of other stocks similarity
        closing_price = np.array(self.stock_c_v_o_h_l[0].T[0])
        closing_price = np.vstack((closing_price, self.stock_c_v_o_h_l[1].T[0]))
        closing_price = np.vstack((closing_price, self.stock_c_v_o_h_l[2].T[0]))
        closing_price = np.float64(closing_price)

        apple_cor = StockCorrelation(closing_price)

        return apple_cor.get_similar_stock_list(self.apple_index)

    def create_params(self, number=1):

        if 0 <= number < self.num_stocks:
            number = number
        else:
            number = 1

        similar_list = self.calc_stock_cor_for_apple()
        return_params = self.stock_c_v_o_h_l[0]
        for i in range(0, number):
            return_params = np.column_stack((return_params, self.stock_c_v_o_h_l[similar_list[i]]))

        # add in class data for apple
        return return_params

    def get_class(self):
        return self.apple_class
