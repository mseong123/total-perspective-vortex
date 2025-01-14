from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
import pandas as pd


def main():

    X = pd.DataFrame(np.array([[8,3,-10], [1,1,-2], [2,5,-1], [4,3,-2],[1,10,-1], [0,1,-10]]), columns=["first", "second","third"])
    y = np.array([2,1,0,0,1,2])
    # clf = tree.DecisionTreeClassifier(criterion="gini")
    # clf.fit(X,y)
    # print(clf.tree_.threshold)
    # print(clf.tree_.impurity)
    # tree.plot_tree(clf)
    # plt.show()
    clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=5)
    clf.fit(X, y)
    

main()