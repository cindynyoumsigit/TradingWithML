""""""
"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Cindy Nyoumsi (replace with your name)
GT User ID: snyoumsi3 (replace with your User ID)
GT ID: 903647082 (replace with your GT ID)
"""
import os
import numpy as np
import pandas as pd
import datetime as dt
from util import get_data, plot_data
import TheoreticallyOptimalStrategy as tos


def author():
	return 'snyoumsi3'

def compute_portvals(symbols, df_trades, start_val=100000,commission=0,impact=0,):
	# ------------------ Code by Cindy ---------------------------------

	print()
	print("df_trades passed in compute_portvals ")
	print(df_trades.head())
	print()
	# ------------------ Building Prices DataFrame
	# obtain list of symbols from df_trades
	orders_symbols = symbols
	sd = df_trades.index.min()
	ed = df_trades.index.max()
	orders_dates = pd.date_range(sd,ed)
	# Build prices dataframe using get_data
	prices_data = get_data(orders_symbols, orders_dates, addSPY = False)
	prices_data = prices_data.dropna()
	prices_data['Cash'] = 1.0
	# print('---------- Prices Data --------')
	# print(prices_data.head())
	# print(prices_data.tail())
	# print()

	# ------------------ Building Trades DataFrame
	trades_data = df_trades.copy()
	trades_data['Cash'] = trades_data[symbols[0]]*-1*prices_data[symbols[0]]
	# print('---------- Trades Data --------')
	# print(trades_data.head())
	# print(trades_data.tail())
	# print()


	# ------------------ Building Holdings DataFrame
	holdings_data = pd.DataFrame(0.0, columns=trades_data.columns, index=trades_data.index)
	holdings_data.loc[sd]['Cash'] = start_val
	holdings_data = holdings_data + trades_data
	holdings_data = holdings_data.cumsum()
	# print('------------- Holdings Data ---------')
	# print(holdings_data.head())
	# print(holdings_data.tail())
	# print()

	# ------------------ Building Values DataFrame
	values_data = prices_data * holdings_data
	# print('--------- Values Data ---------')
	# print(values_data.head())
	# print()

	# ------------------ Building Portfolio Values DataFrame
	portval_data = pd.DataFrame(0.0, columns=['portfolio_value'], index=values_data.index)
	portval_data['portfolio_value'] = values_data.sum(axis=1)
	# print('--------- Portfolio Values Data ---------')
	# print(portval_data.head())
	# print()

	portvals = portval_data
	return portvals


def portfolio_stats(port_val):
	# add code here to compute stats
	dr = (port_val / port_val.shift(1)) - 1
	cr = (port_val.iloc[-1,0]/port_val.iloc[0,0]) - 1
	adr = dr.mean()
	sddr = dr.std()

	return  cr, adr, sddr


def test_code():
	pass


if __name__ == "__main__":
	test_code()
