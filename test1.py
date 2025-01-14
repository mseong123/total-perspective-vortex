
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():

    X = np.array([[8,3,-10], [1,1,-2], [2,5,-1], [4,3,-2],[1,10,-1], [0,1,-10]])
    y = np.array([0,1,0,1,0,1])
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=1)
    clf.fit(X,y)
    print("n_node_samples",clf.tree_.n_node_samples)
    print("threshold",clf.tree_.threshold)
    print("impurity",clf.tree_.impurity)
    print("feature",clf.tree_.feature)
    tree.plot_tree(clf)
    # plt.show()
    

main()