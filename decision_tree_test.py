'''to test and benchmark own implementation of decision tree classifier vs scikit learn's classifier'''

from sklearn.tree import DecisionTreeClassifier
from decision_tree import DecisionTreeClassifier as MyDecisionTreeClassifier
import numpy as np

def main():
    '''main'''
    clf = DecisionTreeClassifier()
    myclf = MyDecisionTreeClassifier()

    X:np.array = np.random.randint(100, size=(100,10))
    y:np.array = np.random.randint(10, size=(100))
    clf.fit(X,y)
    myclf.fit(X,y)
    print("Scikit Learn decision tree classifier prediction")
    print("-----------------------------------")
    print("prediction", clf.predict(X))
    print("accuracy score", clf.score(X,y))
    print("-----------------------------------")
    print("Own decision tree classifier prediction")
    print("-----------------------------------")
    print("prediction", myclf.predict(X))
    print("accuracy score", myclf.score(X,y))
    print("-----------------------------------")



if __name__ == "__main__":
    main()
