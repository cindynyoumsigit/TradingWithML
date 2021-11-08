""""""
"""MC2-P6: Indicator Evaluator.

-----do not edit anything above this line---

Student Name: Cindy Nyoumsi (replace with your name)
GT User ID: snyoumsi3 (replace with your User ID)
GT ID: 903647082 (replace with your GT ID)
"""

import os
import util as ut
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import random as rand
import RTLearner as rt
import BagLearner as bl
import indicators as indic
import marketsimcode as mkt
import matplotlib.pyplot as plt
sns.set()


class StrategyLearner(object):

    def author():
        return 'snyoumsi3'

    # constructor
    def __init__(self, verbose = False, impact=0.0, commission = 0.00):
        """
        Constructor method
        """
        self.ldays = 5
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {"leaf_size": 10}, bags = 20, boost = False, verbose = False)

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # add your code to do learning here

        # example usage of the old backward compatible util function
        # syms = [symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Cindy's Learning Code
        # Get Prices Data
        orders_symbols = [symbol]
        orders_dates = pd.date_range(sd, ed)
        prices_data = ut.get_data(orders_symbols, orders_dates)
        prices_data = prices_data[orders_symbols]
        # prices_SPY = prices_data["SPY"]  # only SPY, for comparison later
        prices_data = prices_data/prices_data.iloc[0,:]
        prices_data = prices_data.fillna(method='ffill')
        prices_data = prices_data.fillna(method='bfill')
        if self.verbose:
            print(prices_data)

        # Calculate indicators
        sma = indic.stock_sma(prices_data)
        percentb = indic.stock_percentb(prices_data, sma)
        moment = indic.stock_momentum(prices_data)
        vol = indic.stock_volatility(prices_data)

        sma_df = sma.rename(columns={symbol:'SMA'})
        percentb_df = percentb.rename(columns={symbol:'PERCENTB'})
        moment_df = moment.rename(columns={symbol:'MOMENT'})
        vol_df = vol.rename(columns={symbol:'VOL'})

        # Build trainX Data
        xtrain_data = pd.concat((percentb_df,moment_df,vol_df),axis=1)
        xtrain_data.fillna(0,inplace=True)
        xtrain_data = xtrain_data[:-self.ldays] # All but the last 5 rows
        xtrain_data = xtrain_data.values

        # Create trade threshold
        threshold = ((prices_data.values[self.ldays:]/prices_data.values[:-self.ldays])-1).T[0]
        buy_thres = 0.02 + self.impact + self.commission
        sell_thres = -0.02 - self.impact - self.commission

        # Build trainY Data
        buy_y = (threshold > buy_thres).astype(int)
        sell_y = (threshold < sell_thres).astype(int)
        ytrain_data = buy_y - sell_y
        ytrain_data = np.array(ytrain_data)

        # Train X and Y datasets
        self.learner.add_evidence(xtrain_data, ytrain_data)


    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """


        # Get Prices Data
        orders_symbols = [symbol]
        orders_dates = pd.date_range(sd, ed)
        prices_data = ut.get_data(orders_symbols, orders_dates)
        prices_data = prices_data[orders_symbols]
        prices_data = prices_data/prices_data.iloc[0,:]
        prices_data = prices_data.fillna(method='ffill')
        prices_data = prices_data.fillna(method='bfill')
        if self.verbose:
            print(prices_data)

        # Calculate indicators
        sma = indic.stock_sma(prices_data)
        percentb = indic.stock_percentb(prices_data, sma)
        moment = indic.stock_momentum(prices_data)
        vol = indic.stock_volatility(prices_data)

        sma_df = sma.rename(columns={symbol:'SMA'})
        percentb_df = percentb.rename(columns={symbol:'PERCENTB'})
        moment_df = moment.rename(columns={symbol:'MOMENT'})
        vol_df = vol.rename(columns={symbol:'VOL'})

        # Build X test Data
        xtest_data = pd.concat((percentb_df,moment_df,vol_df),axis=1)
        xtest_data.fillna(0,inplace=True)
        xtest_data = xtest_data.values

        # Build Y test Data using learner
        ytest_data = self.learner.query(xtest_data)
        ytest_data = np.array(ytest_data)

        # Building Trades Dataframe
        trades_data = np.zeros(ytest_data.shape[0])
        curr_pos = 0
        for i in range(ytest_data.shape[0]-1):
            if(ytest_data[i]>0):
                if(curr_pos ==0):
                    trades_data[i]= 1000
                elif(curr_pos == -1):
                    trades_data[i]= 2000
                curr_pos = 1
            if(ytest_data[i]<0):
                if(curr_pos == 0):
                    trades_data[i]= -1000
                elif(curr_pos ==1):
                    trades_data[i]= -2000
                curr_pos = -1

        if(curr_pos ==-1):
            trades_data[-1]=1000
        elif(curr_pos ==1):
            trades_data[-1]=-1000

        trades_data=pd.DataFrame(trades_data, index=prices_data.index)
        trades_data.columns = orders_symbols

        return trades_data


if __name__ == "__main__":
    print("One does not simply think up a strategy")
    # learner = StrategyLearner(verbose = False, impact = 0.0005, commission = 9.95) # constructor
    learner = StrategyLearner()
    learner.add_evidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase
    print(df_trades)
    print(np.count_nonzero(df_trades))
