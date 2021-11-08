""""""
"""MC2-P6: Test Project Evaluator.

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
import indicators as indic
import ManualStrategy as ms
import marketsimcode as mkt
import StrategyLearner as sl
import matplotlib.pyplot as plt
from util import get_data, plot_data
sns.set()


def author():
	return 'snyoumsi3'

def test_code():
	# Create Parameters
	symbols = ['JPM']
	sd = dt.datetime(2008, 1, 1)
	ed = dt.datetime(2009, 12, 31)
	orders_dates = pd.date_range(sd, ed)

	# Set up for StrategyLearner
	learner = sl.StrategyLearner()
	learner.add_evidence(symbol = "JPM", sd = sd, ed = ed, sv = 100000)

	# Compute Df_trades
	ms_df_trades = ms.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
	sl_df_trades = learner.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
	benchmark_trades = ms.benchmarkPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)

	# Compute Portfolio values
	ms_portvals = mkt.compute_portvals(symbols, ms_df_trades, start_val=100000)
	sl_portvals = mkt.compute_portvals(symbols, sl_df_trades, start_val=100000)
	benchmark_portvals = mkt.compute_portvals(symbols, benchmark_trades, start_val=100000)

	# Normalize Portfolio Values Data
	ms_portvals = ms_portvals/ms_portvals.iloc[0,:]
	sl_portvals = sl_portvals/sl_portvals.iloc[0,:]
	benchmark_portvals = benchmark_portvals/benchmark_portvals.iloc[0,:]

	# Get portfolio stats
	ms_cum_ret, ms_avg_daily_ret, ms_std_daily_ret = mkt.portfolio_stats(ms_portvals)
	print()
	print()
	print('------------- Manual Strategy Portfolio Metrics  ---------')
	print(f"Cumulative Return of MS = %.4f" % ms_cum_ret)
	print(f"Standard Deviation of Daily Returns of MS =  %.4f"% ms_std_daily_ret)
	print(f"Average Daily Returns  of MS =  %.4f"% ms_avg_daily_ret)
	print()
	sl_cum_ret, sl_avg_daily_ret, sl_std_daily_ret = mkt.portfolio_stats(sl_portvals)
	print()
	print()
	print('------------- Strategy Learner Portfolio Metrics  ---------')
	print(f"Cumulative Return of SL = %.4f" % sl_cum_ret)
	print(f"Standard Deviation of Daily Returns of SL =  %.4f"% sl_std_daily_ret)
	print(f"Average Daily Returns  of SL =  %.4f"% sl_avg_daily_ret)
	print()
	bench_cum_ret, bench_avg_daily_ret, bench_std_daily_ret = mkt.portfolio_stats(benchmark_portvals)
	print()
	print('------------- Benchmark Portfolio Metrics ---------')
	print(f"Cumulative Return of Benchmark =  %.4f" % bench_cum_ret)
	print(f"Standard Deviation of Daily Returns of Benchmark =  %.4f" % bench_std_daily_ret)
	print(f"Average Daily Returns of Benchmark =  %.4f"% bench_avg_daily_ret)
	print()

	# Chart Creation
	ax = ms_portvals.plot(color = 'pink')
	sl_portvals.plot(ax = ax, color = 'orange')
	benchmark_portvals.plot(ax = ax, color = 'green')
	plt.xlabel('Date')
	plt.ylabel('Normalized Portfolio Values')
	plt.legend(["Normalized MS","Normalized SL", "Normalized Benchmark"])
	plt.title('Normalized MS, "Normalized SL" and Benchmark')
	plt.savefig('exp1.png')
	plt.close()





if __name__ == "__main__":
	test_code()
