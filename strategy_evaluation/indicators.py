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
import matplotlib.pyplot as plt
from util import get_data, plot_data
import seaborn as sns
sns.set()

def author():
	return 'snyoumsi3'

# Indicator #1
def stock_sma(normed_prices_data, roll_window=20):
	# SMA (Simple Moving Average) calculation
	sma = normed_prices_data.rolling(window=roll_window).mean()
	# SMA Chart Creation
	ax = normed_prices_data.plot()
	sma.plot(ax = ax)
	plt.xlabel('Date')
	plt.ylabel('Normalized Stock Prices')
	plt.legend(["Normalized Stock Prices", "SMA"])
	plt.title('Normalized Price Data and SMA')
	plt.savefig('sma.png')
	plt.close()
	return sma

# Indicator #2
def stock_bbp(normed_prices_data, sma, roll_window=20):
	# BBP ( Bollinger Bands) calculation
	rolling_std = normed_prices_data.rolling(window=roll_window, min_periods=roll_window).std()
	upper_band = sma + (2 * rolling_std)
	lower_band = sma - (2 * rolling_std)
	# BBP Chart Creation
	ax = normed_prices_data.plot()
	upper_band.plot(ax = ax)
	lower_band.plot(ax = ax)
	sma.plot(ax = ax)
	plt.xlabel('Date')
	plt.ylabel('Normalized Stock Prices')
	plt.legend(["Normalized Stock Prices", "Upper Band", "Lower Band","SMA","%B"])
	plt.title('Normalized Price Data and BBP')
	plt.savefig('bbp.png')
	plt.close()
	return stock_bbp

# Indicator #3
def stock_percentb(normed_prices_data, sma, roll_window=20):
	# BBP ( Bollinger Bands) calculation
	rolling_std = normed_prices_data.rolling(window=roll_window, min_periods=roll_window).std()
	upper_band = sma + (2 * rolling_std)
	lower_band = sma - (2 * rolling_std)
	percent_b = (normed_prices_data - lower_band)/(upper_band - lower_band)
	# %B Chart Creation
	ax = normed_prices_data.plot()
	sma.plot(ax = ax)
	percent_b.plot(ax = ax)
	plt.xlabel('Date')
	plt.ylabel('Normalized Stock Prices')
	plt.legend(["Normalized Stock Prices","SMA","%B"])
	plt.title('Normalized Price Data, SMA and %B')
	plt.savefig('percentb.png')
	plt.close()
	return percent_b

# Indicator #4
def stock_momentum(normed_prices_data, roll_window=20):
	# Momentum calculation
	momentum = (normed_prices_data/normed_prices_data.shift(roll_window)) - 1
	momentum_avg = momentum.rolling(window=roll_window).mean()
	# Momentum Chart Creation
	ax = normed_prices_data.plot()
	momentum.plot(ax = ax)
	momentum_avg.plot(ax = ax)
	plt.xlabel('Date')
	plt.ylabel('Normalized Stock Prices')
	plt.legend(["Normalized Stock Prices", "Momentum","Momentum Average"])
	plt.title('Normalized Price Data and Momentum')
	plt.savefig('momentum.png')
	plt.close()
	return momentum

# Indicator #5
def stock_volatility(normed_prices_data, roll_window=20):
	# Volatility calculation
	volatility = 3.5*(normed_prices_data.rolling(window=roll_window, min_periods=roll_window).std())
	# volatility Chart Creation
	ax = normed_prices_data.plot()
	volatility.plot(ax = ax)
	plt.xlabel('Date')
	plt.ylabel('Normalized Stock Prices')
	plt.legend(["Normalized Stock Prices", "Volatility"])
	plt.title('Normalized Price Data and volatility')
	plt.savefig('volatility.png')
	plt.close()
	return volatility


def test_code():
	symbols = ['JPM']
	start_date = dt.datetime(2008, 1, 1)
	end_date = dt.datetime(2009, 12, 31)
	dates = pd.date_range(start_date, end_date)
	prices_data = get_data(symbols, dates)
	prices_data = prices_data[symbols]
	normed_prices_data = prices_data/prices_data.iloc[0,:]

	sma = stock_sma(normed_prices_data)
	bbp = stock_bbp(normed_prices_data, sma)
	percentb = stock_percentb(normed_prices_data, sma)
	moment = stock_momentum(normed_prices_data)
	vol = stock_volatility(normed_prices_data)

if __name__ == "__main__":
	test_code()
