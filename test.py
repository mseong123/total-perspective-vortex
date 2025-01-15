from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTreeClassifier
import pandas as pd


def main():

    X = pd.DataFrame(np.array([[2,3,], [1,1], [3,5], [4,3],[1,10], [0,1]]), columns=["first", "second"])
    y = np.array([1,0,1,1,2,1])
    # clf = tree.DecisionTreeClassifier(criterion="gini")
    # clf.fit(X,y)
    # print(clf.tree_.threshold)
    # print(clf.tree_.impurity)
    # tree.plot_tree(clf)
    # plt.show()
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    print(clf.predict(X))
    

main()