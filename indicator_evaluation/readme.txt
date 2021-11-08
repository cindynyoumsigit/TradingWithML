# READ ME TXT

## Description of what each python file is for/does.

File: indicators.py
What it does: It contains indicator functions that each return dataframes of the indicators and their charts.

File: marketsimcode.py
What it does: It accepts a single column “trades” DataFrame and returns a single column portfolio value dataframe and portfolio statistics.

File: TheoreticallyOptimalStrategy.py
What it does: It returns a  a single column “trades” DataFrame and then calls on marketsimcode.py to compute the portfolio value using the trades it returns.

File: testproject.py
What it does: This file is used to run the files: indicators.py , TheoreticallyOptimalStrategy.py  and marketsimcode.py  to generate portfolio statistics and charts. 

## Explicit instructions on how to properly run your code.
Use the testproject.py file and run it using the command: PYTHONPATH=../:. python testproject.py
