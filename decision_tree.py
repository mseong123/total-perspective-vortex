'''Own implementation of decision tree classifier base on Scikit learn prototype implementing 
hyperparams criterion, min_samples_leaf and max_depth'''

import math
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Tree_():
    def __init__(self):
        # array of shape (node, gini value for current node)
        self.threshold:np.array = np.array([])
        # array of shape (node, no of samples for current node)
        self.n_node_samples:np.array = np.array([])
        # array of shape (node, list of value). Element in value list = no. of samples for each
        # class for current node)
        self.value:np.array = np.array([])

class DecisionTreeClassifier(BaseEstimator, TransformerMixin):
    '''class definition'''
    def __init__(self, criterion:str='gini', min_samples_split:int = 2, max_depth:int | None = None):
        self.criterion:str = criterion
        self.min_samples_split:int = min_samples_split
        self.max_depth:int | None = max_depth
        self.n_classes:np.array = np.array([])
        self.depth:float = 0
        # Tree_ class instance to hold tree information in each node (and leaf), tree_.<attribute>[0] is root,
        # index is according to depth first traversal of the node from root.
        tree_:Tree_ = Tree_()

    def gini_index(self, y_segment:pd.Series) -> float:
        '''calculate gini for segment of sample'''
        # get unique labels
        classes:np.array = y_segment.unique()
        result:float = 1
        # prob dist per class
        prob_class:list[float] = [(y_segment == value).sum().item() / len(y_segment) for value in classes.tolist()]
        # calculate entropy for this node
        for prob in prob_class:
            result = result - (prob ** 2)
        return result
    
    def entropy(self, y_segment:pd.Series) -> float:
        '''calculate information gain(entropy) for segment of sample'''
        # get unique labels
        classes:np.array = y_segment.unique()
        result:float = 0
        # calculate entropy for this node
        for value in classes.tolist():
            prob:float = (y_segment == value).sum().item() / len(y_segment)
            result = result - ((prob) * math.log2(prob))
        return result



    def partition(self, X_segment:pd.DataFrame, y_segment:pd.Series, depth:int, threshold:float, )-> None:
        '''recursive function to partition dataset into decision tree as per criteria (gini or entropy)'''
        if depth > self.depth:
            self.depth = depth
        np.append(self.tree_.n_node_samples, len(y_segment))
        np.append(self.tree_.threshold, threshold)
        classes:np.array = y_segment.unique()
        value:np.array = np.array([(y_segment == value).sum().item() for value in classes])
        np.append(self.tree_.value)
        if len(X_segment) != 1 and len(X_segment) >= self.min_samples_split and depth < self.max_depth:
            




    def fit(self, X:pd.DataFrame | np.array, y:pd.Series | np.array):
        if isinstance(X, pd.DataFrame) == False:
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series) == False:
            y = pd.Series(y)
        self.n_classes:np.array = y.unique()
        self.partition(X, y, 0)
        return self