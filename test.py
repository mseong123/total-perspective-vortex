from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt


def main():
    X = np.array([[0,0], [1,1], [5,3], [4,3],[3,4], [0,1]])
    y = np.array([0,1,0,1,0,1])
    clf = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=2)
    clf.fit(X,y)
    print(clf.tree_.weighted_n_node_samples)
    print(clf.tree_.n_node_samples)
    print(clf.tree_.impurity)
    tree.plot_tree(clf)
    plt.show()
    

main()