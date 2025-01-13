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
    clf = tree.DecisionTreeClassifier(criterion="gini")
    clf.fit(X,y)
    print(type(clf.tree_.n_node_samples))
    print(clf.tree_.n_node_samples)
    print(clf.tree_.value)
    tree.plot_tree(clf)
    plt.show()
    # clf = DecisionTreeClassifier()
    # clf.fit(X, y)
    

main()