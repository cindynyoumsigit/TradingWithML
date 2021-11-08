""""""
"""
Test a learner.  (c) 2015 Tucker Balch

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
"""

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import time as time

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array(
        [list(map(str, s.strip().split(","))) for s in inf.readlines()]
    )
    # remove header and date column
    data = data[1:,1:]
    # Convert back to float
    data = data.astype(np.float)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # -------------------------------- Create and Train DTlearner ---------------

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner = dt.DTLearner(1, verbose=True)  # DTLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")

    # ------- Experiment 1 and 3.1
    # initialize variables
    dt_insample_rmse_list = []
    dt_outsample_rmse_list = []
    dt_time_list = []
    dt_insample_std_list = []
    dt_outsample_std_list = []
    max_leafsize = 100

    # Run DTLearner with different leaf_sizes
    for leafsize in range(max_leafsize):
        start_time = time.time()
        learner = dt.DTLearner(leafsize, verbose=True)
        learner.add_evidence(train_x, train_y)
        end_time = time.time()
        time_taken = end_time - start_time
        dt_time_list.append(time_taken)

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        dt_insample_rmse_list.append(rmse)
        dt_std = np.std(pred_y)
        dt_insample_std_list.append(dt_std)

        # evaluate out of sample rmse & std
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        dt_outsample_rmse_list.append(rmse)
        dt_std = np.std(pred_y)
        dt_outsample_std_list.append(dt_std)

    # Plot rmse against leaf_size
    # insample_rmse_list.plot()
    # outsample_rmse_list.plot()
    plt.plot(dt_insample_rmse_list)
    plt.plot(dt_outsample_rmse_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend(["Train RMSE", "Test RMSE"])
    plt.title('DTLearner RMSE using different Leaf Sizes')
    plt.savefig('Experiment1.png')
    plt.close()

    # -------------------------------- Create and Train RTlearner ---------------

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner = rt.RTLearner(1, verbose=True)  # RTLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")

    # ------- Experiment 3.2
    # initialize variables
    rt_insample_rmse_list = []
    rt_outsample_rmse_list = []
    rt_time_list = []
    rt_insample_std_list = []
    rt_outsample_std_list = []
    max_leafsize = 100

    # Run DTLearner with different leaf_sizes
    for leafsize in range(max_leafsize):
        start_time = time.time()
        learner = rt.RTLearner(leafsize, verbose=True)
        learner.add_evidence(train_x, train_y)
        end_time = time.time()
        time_taken = end_time - start_time
        rt_time_list.append(time_taken)

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rt_insample_rmse_list.append(rmse)
        rt_std = np.std(pred_y)
        rt_insample_std_list.append(rt_std)

        # evaluate out of sample rmse & std
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        rt_outsample_rmse_list.append(rmse)
        rt_std = np.std(pred_y)
        rt_outsample_std_list.append(rt_std)

    # Plot time against leaf_size
    plt.plot(dt_time_list)
    plt.plot(rt_time_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('Time')
    plt.legend(["DT Learner", "RT Learner"])
    plt.title('Time Performance comparison for DT & RT learners with different Leaf Sizes')
    plt.savefig('Experiment31.png')
    plt.close()

    # Plot in sample DT & RT std against leaf_size
    plt.plot(dt_insample_std_list)
    plt.plot(rt_insample_std_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('Standard Deviation of Predicted values')
    plt.legend(["DT In Sample STD", "RT In Sample STD"])
    plt.title('In Sample STD comparison for DT & RT learners with different Leaf Sizes')
    plt.savefig('Experiment32.png')
    plt.close()

    # Plot out of sample DT & RT std against leaf_size
    plt.plot(dt_outsample_std_list)
    plt.plot(rt_outsample_std_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('Standard Deviation of Predicted values')
    plt.legend(["DT Out Sample STD", "RT Out Sample STD"])
    plt.title('Out of Sample STD comparison for DT & RT learners with different Leaf Sizes')
    plt.savefig('Experiment33.png')
    plt.close()

    # -------------------------------- Create and Train BagLearnerlearner ---------------

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)  # BagLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")

    # ------- Experiment 2
    # initialize variables
    insample_rmse_list = []
    outsample_rmse_list = []
    max_leafsize = 100

    # Run BagLearner with different leaf_sizes
    for leafsize in range(max_leafsize):
        learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":leafsize}, bags = 20, boost = False, verbose = False)  # BagLearner
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = learner.query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        insample_rmse_list.append(rmse)

        # evaluate out of sample
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        outsample_rmse_list.append(rmse)

    # Plot rmse against leaf_size
    # insample_rmse_list.plot()
    # outsample_rmse_list.plot()
    plt.plot(insample_rmse_list)
    plt.plot(outsample_rmse_list)
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend(["Train RMSE", "Test RMSE"])
    plt.title('20 Bags BagLearner RMSE using different Leaf Sizes')
    plt.savefig('Experiment2.png')
    plt.close()

    # -------------------------------- Create and Train InsaneLearner ---------------

    # create a learner and train it
    # learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner = it.InsaneLearner(verbose = False)  # InsaneLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
