# READ ME TXT

## Description of what each python file for my strategy evaluation project is for/does.


File 1: RTLearner.py
What it does: It contains the implemented Classification Learner I am using for my StrategyLearner

File 2: BagLearner.py
What it does: It contains the bagging implementation I am using for my StrategyLearner

File 3: ManualStrategy.py
What it does: It contains the Manual Strategy I implemented using if-else conditions on my indicators for trading

File 4: StrategyLearner.py
What it does: It contains the implementation of the Machine Learning Classification model I am using for trading


File 5: indicators.py
What it does: It contains indicator functions that each return dataframes of the indicators and their charts.

File 6: marketsimcode.py
What it does: It accepts a single column “trades” DataFrame and returns a single column portfolio value dataframe and portfolio statistics.

File 7: experiment1.py
What it does: It computes portfolio values, portfolio statistics, and generates charts to compare between the benchmark policy, manual strategy and strategy learner.

File 8: experiment2.py
What it does: It computes portfolio values using StrategyLearner at different impact levels and generates charts for comparison.

File 9: testproject.py
What it does: This file is used to run the files listed above.

## Explicit instructions on how to properly run your code.
Use the testproject.py file and run it using the command: PYTHONPATH=../:. python testproject.py
