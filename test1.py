
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    X = np.array([[2,3,], [8,7], [3,5], [4,3],[1,10], [0,1]])
    # X = np.array([[15,3], [1,15], [3,5], [4,3],[1,10], [0,1], [2,5],[5,1],[9,2],[2,2],[3,2]])
    # y = np.array([1,1,0,1,0,1,0,1,0,1,0])
    y = np.array([1,0,1,1,2,1])
    clf = tree.DecisionTreeClassifier(criterion="gini")
    clf.fit(X,y)
    print("n_node_samples",clf.tree_.n_node_samples)
    print("threshold",clf.tree_.threshold)
    print("impurity",clf.tree_.impurity)
    print("feature",clf.tree_.feature)
    print("children_left", clf.tree_.children_left)
    print("children_right", clf.tree_.children_right)

    tree.plot_tree(clf)
    # plt.show()
    print(clf.predict(X))
    

main()