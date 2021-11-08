""""""
"""MC2-P6: Test Project Evaluator.

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
import indicators as indic
import marketsimcode as mkt
import TheoreticallyOptimalStrategy as tos
import seaborn as sns
sns.set()


def author():
	return 'snyoumsi3'

if __name__ == "__main__":
	indic.test_code()
	tos.test_code()
