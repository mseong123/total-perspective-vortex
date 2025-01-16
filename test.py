from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
import pandas as pd


def main():

    X = pd.DataFrame(np.array([[2,3,], [8,7], [3,5], [4,3],[1,10], [0,1]]), columns=["first", "second"])
    # X = pd.DataFrame(np.array([[15,3], [1,15], [3,5], [4,3],[1,10], [0,1], [2,5],[5,1],[9,2],[2,2],[3,2]]), columns=["first", "second"])
    y = np.array([1,0,1,1,2,1])
    # y = np.array([1,1,0,1,0,1,0,1,0,1,0])
    # clf = tree.DecisionTreeClassifier(criterion="gini")
    # clf.fit(X,y)
    # print(clf.tree_.threshold)
    # print(clf.tree_.impurity)
    # tree.plot_tree(clf)
    # plt.show()
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(X, y)
    print(clf.predict(X))
    

main()