""""""
"""MC2-P6: Indicator Evaluator.

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

	# Build Orders data based on future prices and following portfolio constraints
	trades_data = prices_data.copy()
	trades_data['Order'] = prices_data < prices_data.shift(-1)
	trades_data['Order'].replace(True, 2000, inplace=True)
	trades_data['Order'].replace(False, -2000, inplace=True)
	trades_data['Hold'] = trades_data['Order'] == trades_data['Order'].shift(+1)
	trades_data['Order'] = np.where(trades_data['Hold']  == True, 0, trades_data['Order'])
	trades_data['Order'][0] = np.where(trades_data['Order'][0] == -2000, -1000, 1000)
	trades_data[[orders_symbols[0]]] = trades_data['Order']
	trades_data = trades_data[[orders_symbols[0]]]

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
	print('------------- TOS Portfolio Metrics  ---------')
	print(f"Cumulative Return of TOS = %.4f" % cum_ret)
	print(f"Standard Deviation of Daily Returns of TOS =  %.4f"% std_daily_ret)
	print(f"Average Daily Returns  of TOS =  %.4f"% avg_daily_ret)
	print()
	bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret = mkt.portfolio_stats(benchmark_portvals)
	print()
	print('------------- Benchmark Portfolio Metrics ---------')
	print(f"Cumulative Return of Benchmark =  %.4f" % bench_cum_ret)
	print(f"Standard Deviation of Daily Returns of Benchmark =  %.4f" % bench_std_daily_ret)
	print(f"Average Daily Returns of Benchmark =  %.4f"% bench_avg_daily_ret)
	print()

	# Chart Creation
	ax = portvals.plot(color = 'r')
	benchmark_portvals.plot(ax = ax, color = 'g')
	plt.xlabel('Date')
	plt.ylabel('Normalized Stock Prices')
	plt.legend(["Normalized TOS", "Normalized Benchmark"])
	plt.title('Normalized TOS and Benchmark')
	plt.savefig('tos.png')
	plt.close()

if __name__ == "__main__":
	test_code()
