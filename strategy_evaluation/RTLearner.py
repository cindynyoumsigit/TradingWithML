""""""
"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch

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

import numpy as np
from scipy import stats

class RTLearner(object):
    """
    This is a Decision Tree Learner. It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size, verbose=False):
        """
        Constructor method
        """
        self.tree = None
        self.leaf_size = leaf_size
        # pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "snyoumsi3"  # replace tb34 with your Georgia Tech username

    # Creating build tree function:

    def bestfeature(self, data):

        # split data into features vs y values to be predicted
        data_x = data[:, 0:data.shape[1]-1]
        # print('data_x  is')
        # print(data_x)
        data_y = data[:, data.shape[1]-1]
        # print('data_y  is')
        # print(data_y)
        # print()

        # initialize list to store all correlation variables
        corr_list = []

        # Loop through every feature to find each correlation value
        # import pdb; pdb.set_trace()
        for feat in range(data_x.shape[1]):
            corr_val = np.corrcoef(data_x[:, feat], data_y)
        #             print('correlation_val = ',corr_val)
        #             print()
            corr_val = abs(corr_val[0, 1])
        #             print('abs correlation_val = ',corr_val)
        #             print()
            corr_list.append(corr_val)

        # print('corr_list is: ' ,corr_list)
        # print()
        max_corr_val = max(corr_list)
        # print('max_corr_val  is: ', max_corr_val)
        # print()
        best_feat = corr_list.index(max_corr_val)
        # print('best_feat is: ', best_feat)
        # print()

        return int(best_feat)

    def build_tree(self, data):

        # Check if we are at the end of the tree and if yes return the average of the leaf values
        if data.shape[0] <= self.leaf_size:
            return np.array([["leaf", np.mean(data[:, -1]), -1, -1]])

        # Check if all the data is the same and if yes return any row value
        if np.all(data[0, -1] == data[:, -1], axis=0):
            return np.array([["leaf", data[0, -1], -1, -1]])

        else:
            # best_feat = self.bestfeature(data)
            best_feat = np.random.randint(data.shape[1]-1)
            split_val = np.median(data[:, best_feat])
            # split_val = np.mean(data[np.random.randint(data.shape[0]), best_feat] + data[np.random.randint(data.shape[0]), best_feat])
            max_val = max(data[:, best_feat])
            if max_val == split_val:
                return np.array([['leaf', np.mean(data[:, -1]), -1, -1]])

        left_tree = self.build_tree(data[data[:, best_feat] <= split_val])
        right_tree = self.build_tree(data[data[:, best_feat] > split_val])

        root = np.array([[best_feat, split_val, 1, left_tree.shape[0]+1]])
        root_left_tree = np.append(root, left_tree, axis=0)

        return np.append(root_left_tree, right_tree, axis=0)

    def tree_ans(self, data_point):
        # initialize node we're looking at
        node = 0

        # Loop through tree until we reach a leaf
        while(self.tree[node, 0] != 'leaf'):

            feat = self.tree[node, 0]
            split_val = self.tree[node, 1]

            # Check condition to identify direction of movement down the tree
            if data_point[int(float(feat))] <= float(split_val):
                # Move to left_tree
                node += int(float(self.tree[node, 2]))

            else:
                # Move to right_tree
                node += int(float(self.tree[node, 3]))

        y_pred_point = self.tree[node, 1]
        return y_pred_point

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """


        data_y = np.array([data_y])
        data_y = data_y.T
        data = np.append(data_x, data_y, axis=1)
        self.tree = self.build_tree(data)

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        # return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[
        #     -1
        # ]

        # Initialize y_pred list and other variables
        y_pred_list = []
        row_num = points.shape[0]

        # For loop to identify predicted y value for every feature data point
        for row in range(0, row_num):
            y_pred = self.tree_ans(points[row, :])
            y_pred = float(y_pred)
            y_pred_list.append(y_pred)

        return np.array(y_pred_list)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
