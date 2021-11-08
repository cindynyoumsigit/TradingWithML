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
import experiment1 as exp1
import experiment2 as exp2
import indicators as indic
import ManualStrategy as ms
import marketsimcode as mkt
import StrategyLearner as sl
import matplotlib.pyplot as plt
from util import get_data, plot_data
sns.set()


def author():
	return 'snyoumsi3'

if __name__ == "__main__":
	np.random.seed(903647082)
	ms.test_code()
	exp1.test_code()
	exp2.test_code()
