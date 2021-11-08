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

class BagLearner(object):
    """
    This is a BagLearner . It is implemented correctly.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, learner, kwargs = {"argument1":1, "argument2":2}, bags = 20, boost = False, verbose = False):
        """
        Constructor method
        """
        self.bags = bags
        self.learner = learner
        self.learners_list = []

        for bag in range(0, bags):
            self.learners_list.append(learner(**kwargs))

        # pass  # move along, these aren't the drones you're looking for

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "snyoumsi3"  # replace tb34 with your Georgia Tech username

    # Creating build tree function:

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """

        data_index_list = np.linspace(0, data_x.shape[0]-1, data_x.shape[0])
        data_index_list = data_index_list.astype(int)

        for learner in self.learners_list:
            data_index = np.random.choice(data_index_list, data_index_list.size)
            learner.add_evidence(data_x.take(data_index, axis=0), data_y.take(data_index,axis=0))

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

        y_pred_list = []
        for learner in self.learners_list:
            y_pred_list.append(learner.query(points))

        y_pred_array = np.array(y_pred_list)
        final_y_pred = np.mean(y_pred_array, axis=0)

        return final_y_pred.tolist()


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
