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

	list_of_impacts=[0.0005,0.005,0.05]
	list_of_portvals=[]

	for impacts in list_of_impacts:
		learner = sl.StrategyLearner(impact = impacts)
		learner.add_evidence(symbol = "JPM", sd = sd, ed = ed, sv = 100000)
		sl_df_trades = learner.testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
		sl_portvals = mkt.compute_portvals(symbols, sl_df_trades, start_val=100000,impact = impacts)
		list_of_portvals.append(sl_portvals)
		sl_cum_ret, sl_avg_daily_ret, sl_std_daily_ret = mkt.portfolio_stats(sl_portvals)
		print()
		print()
		print('------------- Strategy Learner Portfolio Metrics for Impact =', impacts, '  ---------')
		print(f"Cumulative Return of SL = %.4f" % sl_cum_ret)
		print(f"Standard Deviation of Daily Returns of SL =  %.4f"% sl_std_daily_ret)
		print(f"Average Daily Returns  of SL =  %.4f"% sl_avg_daily_ret)
		print()


	# Normalize Portfolio Values Data
	sl_portvals_0005 = list_of_portvals[0]
	sl_portvals_005 = list_of_portvals[1]
	sl_portvals_05 = list_of_portvals[2]

	sl_portvals_0005 = sl_portvals_0005/sl_portvals_0005.iloc[0,:]
	sl_portvals_005 = sl_portvals_005/sl_portvals_005.iloc[0,:]
	sl_portvals_05 = sl_portvals_05/sl_portvals_05.iloc[0,:]

	# sl_portvals = sl_portvals/sl_portvals.iloc[0,:]

	# Chart Creation
	ax = sl_portvals_05.plot(color = 'green')
	sl_portvals_005.plot(ax = ax, color = 'orange')
	sl_portvals_0005.plot(ax = ax, color = 'blue')
	plt.xlabel('Date')
	plt.ylabel('Normalized Portfolio Values')
	plt.legend(["Impact = 0.0005","Impact = 0.005", "Impact = 0.05"])
	plt.title('In Sample Portfolio Values using different Impact Values')
	plt.savefig('exp2.png')
	plt.close()


if __name__ == "__main__":
	test_code()
