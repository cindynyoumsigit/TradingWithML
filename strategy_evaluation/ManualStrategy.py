""""""
"""MC2-P8: Manual Strategy Learner.

-----do not edit anything above this line---

Student Name: Cindy Nyoumsi (replace with your name)
GT User ID: snyoumsi3 (replace with your User ID)
GT ID: 903647082 (replace with your GT ID)
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import random as rand
import indicators as indic
import marketsimcode as mkt
import matplotlib.pyplot as plt
from util import get_data, plot_data

sns.set()

def author():
    return 'snyoumsi3'

def testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):

    # Get Prices Data
    orders_symbols = [symbol]
    orders_dates = pd.date_range(sd, ed)
    prices_data = get_data(orders_symbols, orders_dates)
    prices_data = prices_data[orders_symbols]
    prices_data = prices_data/prices_data.iloc[0,:]
    prices_data = prices_data.fillna(method='ffill')
    prices_data = prices_data.fillna(method='bfill')

    # Calculate indicators
    sma = indic.stock_sma(prices_data)
    percentb = indic.stock_percentb(prices_data, sma)
    moment = indic.stock_momentum(prices_data)
    vol = indic.stock_volatility(prices_data)

	# Build Orders data based on future prices and following portfolio constraints
    trades_data = prices_data.copy()
    curr_pos = 0

    for row in range(prices_data.shape[0]):
        if curr_pos == 0:
            if (percentb.ix[row,0] <= 0.4) or (moment.ix[row,0] > 0.1) or (vol.ix[row,0] > 0.1):
                trades_data.loc[prices_data.index[row], 'Order'] = 1000
                curr_pos = 1
            elif (percentb.ix[row,0] >= 0.6) or (moment.ix[row,0] < -0.1) or (vol.ix[row,0] < 0.1):
                trades_data.loc[prices_data.index[row], 'Order'] = -1000
                curr_pos = -1

        elif curr_pos == 1:
            if (percentb.ix[row,0] >= 0.8) or (moment.ix[row,0] < -0.2) or (vol.ix[row,0] < 0.2):
                trades_data.loc[prices_data.index[row], 'Order']  = -2000
                curr_pos = -1
            elif (percentb.ix[row,0] >= 0.6) or (moment.ix[row,0] < -0.1) or (vol.ix[row,0] < 0.1):
                trades_data.loc[prices_data.index[row], 'Order']  = -1000
                curr_pos = -1

        elif curr_pos == -1:
            if (percentb.ix[row,0] <= 0.2) or (moment.ix[row,0] > 0.2) or (vol.ix[row,0] > 0.2):
                trades_data.loc[prices_data.index[row], 'Order']  = 2000
                curr_pos = 1
            elif (percentb.ix[row,0] <= 0.4) or (moment.ix[row,0] > 0.1) or (vol.ix[row,0] > 0.1):
                trades_data.loc[prices_data.index[row], 'Order']  = 1000
                curr_pos = 1

    if curr_pos == 1:
        trades_data.loc[prices_data.index[row], 'Order']  = -1000
    if curr_pos == -1:
        trades_data.loc[prices_data.index[row], 'Order']  = 1000

    trades_data = trades_data.drop(columns = orders_symbols)
    trades_data.columns = orders_symbols

    trades_data = trades_data.fillna(0)
    trades_data[orders_symbols[0]][0] = 1000

    return trades_data


def benchmarkPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    # Get Prices Data
    orders_symbols = [symbol]
    orders_dates = pd.date_range(sd, ed)
    prices_data = get_data(orders_symbols, orders_dates)
    prices_data = prices_data[orders_symbols]
    prices_data = prices_data/prices_data.iloc[0,:]
    prices_data = prices_data.fillna(method='ffill')
    prices_data = prices_data.fillna(method='bfill')

    # Build Orders data based on future prices and following portfolio constraints
    trades_data = pd.DataFrame(0.0, columns=orders_symbols, index=prices_data.index)
    trades_data[orders_symbols[0]][0] = 1000

    return trades_data


def test_code():
    # ---------------------------------- In Sample Parameters
    symbols = ['JPM']
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    orders_dates = pd.date_range(sd, ed)

    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    benchmark_trades = benchmarkPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)

    # Get Portfolio values
    portvals = mkt.compute_portvals(symbols, df_trades, start_val=100000)
    benchmark_portvals = mkt.compute_portvals(symbols, benchmark_trades, start_val=100000)

    # Normalize Portfolio Values Data
    portvals = portvals/portvals.iloc[0,:]
    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0,:]

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret = mkt.portfolio_stats(portvals)
    print()
    print()
    print('------------- Manual Strategy Portfolio Metrics  ---------')
    print(f"In Sample Cumulative Return of MS = %.4f" % cum_ret)
    print(f"In Sample Standard Deviation of Daily Returns of MS =  %.4f"% std_daily_ret)
    print(f"In Sample Average Daily Returns  of MS =  %.4f"% avg_daily_ret)
    print()
    bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret = mkt.portfolio_stats(benchmark_portvals)
    print()
    print('------------- Benchmark Portfolio Metrics ---------')
    print(f"In Sample Cumulative Return of Benchmark =  %.4f" % bench_cum_ret)
    print(f"In Sample Standard Deviation of Daily Returns of Benchmark =  %.4f" % bench_std_daily_ret)
    print(f"In Sample Average Daily Returns of Benchmark =  %.4f"% bench_avg_daily_ret)
    print()

    # Chart Creation
    ax = portvals.plot(color = 'r')
    benchmark_portvals.plot(ax = ax, color = 'g')
    short_dates = df_trades[df_trades['JPM']<0].index
    long_dates = df_trades[df_trades['JPM']>0].index
    ax.vlines(short_dates, portvals.min(), portvals.max(), color="black")
    ax.vlines(long_dates, portvals.min(), portvals.max(), color="blue")
    plt.xlabel('Date')
    plt.ylabel('Normalized Stock Prices')
    plt.legend(["Normalized MS", "Normalized Benchmark"])
    plt.title('In Sample Normalized MS and Benchmark')
    plt.savefig('insamplemos.png')
    plt.close()

    # ---------------------------------- Out of Sample Parameters
    symbols = ['JPM']
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    orders_dates = pd.date_range(sd, ed)

    df_trades = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    benchmark_trades = benchmarkPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)

    # Get Portfolio values
    portvals = mkt.compute_portvals(symbols, df_trades, start_val=100000)
    benchmark_portvals = mkt.compute_portvals(symbols, benchmark_trades, start_val=100000)

    # Normalize Portfolio Values Data
    portvals = portvals/portvals.iloc[0,:]
    benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0,:]

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret = mkt.portfolio_stats(portvals)
    print()
    print()
    print('------------- Manual Strategy Portfolio Metrics  ---------')
    print(f"Out of Sample Cumulative Return of MS = %.4f" % cum_ret)
    print(f"Out of Sample Standard Deviation of Daily Returns of MS =  %.4f"% std_daily_ret)
    print(f"Out of SampleAverage Daily Returns  of MS =  %.4f"% avg_daily_ret)
    print()
    bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret = mkt.portfolio_stats(benchmark_portvals)
    print()
    print('------------- Benchmark Portfolio Metrics ---------')
    print(f"Out of Sample Cumulative Return of Benchmark =  %.4f" % bench_cum_ret)
    print(f"Out of Sample Standard Deviation of Daily Returns of Benchmark =  %.4f" % bench_std_daily_ret)
    print(f"Out of Sample  Average Daily Returns of Benchmark =  %.4f"% bench_avg_daily_ret)
    print()

    # Chart Creation
    ax = portvals.plot(color = 'r')
    benchmark_portvals.plot(ax = ax, color = 'g')
    short_dates = df_trades[df_trades['JPM']<0].index
    long_dates = df_trades[df_trades['JPM']>0].index
    ax.vlines(short_dates, portvals.min(), portvals.max(), color="black")
    ax.vlines(long_dates, portvals.min(), portvals.max(), color="blue")
    plt.xlabel('Date')
    plt.ylabel('Normalized Stock Prices')
    plt.legend(["Normalized MS", "Normalized Benchmark"])
    plt.title('Out of Sample Normalized MS and Benchmark')
    plt.savefig('outsamplemos.png')
    plt.close()

if __name__ == "__main__":
    test_code()
