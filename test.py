from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
import pandas as pd


def main():
    # X = pd.DataFrame(np.array([[0,0], [1,1], [5,3], [4,3],[3,4], [0,1]]))
    # y = pd.Series(np.array([0,1,0,1,0,1]))

    X = np.array([[0,0], [1,1], [5,3], [4,3],[3,4], [0,1]])
    y = np.array([0,1,0,1,0,1])
    # clf = tree.DecisionTreeClassifier(criterion="gini")
    # clf.fit(X,y)
    # print(clf.tree_.feature)
    # print(clf.tree_.impurity)
    # tree.plot_tree(clf)
    # plt.show()
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(X, y)
    

main()