
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():

    X = np.array([[0,0], [1,1], [5,3], [4,3],[3,4], [0,1]])
    y = np.array(["a","b","a","b","a","b"])
    clf = tree.DecisionTreeClassifier(criterion="gini")
    clf.fit(X,y)
    print(clf.tree_.n_node_samples)
    print(clf.tree_.impurity)
    tree.plot_tree(clf)
    plt.show()
    

main()