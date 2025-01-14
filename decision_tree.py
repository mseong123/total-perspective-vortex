'''Own implementation of decision tree classifier base on Scikit learn prototype implementing 
hyperparams criterion, min_samples_leaf and max_depth'''

import math
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class Tree_():
    def __init__(self):
        # array for threshold value for node at index
        self.threshold:np.array = np.array([])
        # array for feature for node at index
        self.feature:np.array = np.array([])
        # array of no of samples for node at index
        self.n_node_samples:np.array = np.array([])
        # array of shape (node, value array). element in value array = no. of samples for each
        # class for current node index)
        self.value:np.array = np.array([])
        # array of impurity (gini or entropy) for node at index
        self.impurity:np.array = np.array([])

class DecisionTreeClassifier(BaseEstimator, TransformerMixin):
    '''class definition'''
    def __init__(self, criterion:str='gini', min_samples_split:int = 2, max_depth:int | None = None):
        self.criterion:str = criterion
        self.min_samples_split:int = min_samples_split
        self.max_depth:int | None = max_depth
        self.n_classes:np.array = np.array([])
        self.depth_:float = 0
        self.feature_names_in_:list | None = None
        # Tree_ class instance to hold tree information in each node (and leaf), tree_.<attribute>[0] is root,
        # index is according to depth first traversal of the node from root.
        self.tree_:Tree_ = Tree_()

    def gini_index(self, y_segment:pd.Series) -> float:
        '''calculate gini for segment of sample'''
        result:float = 1
        # prob dist per class
        prob_class:list[float] = [(y_segment == value).sum().item() / len(y_segment) for value in self.classes_.tolist()]
        # calculate entropy for this node
        for prob in prob_class:
            result = result - (prob ** 2)
        return result
    
    def entropy(self, y_segment:pd.Series) -> float:
        '''calculate information gain(entropy) for segment of sample'''
        result:float = 0
        # calculate entropy for this node
        for value in self.classes_.tolist():
            prob:float = (y_segment == value).sum().item() / len(y_segment)
            result = result - ((prob) * math.log2(prob))
        return result
    
    def check_subnode_split(self, X_segment:pd.DataFrame, y_segment:pd.Series) -> tuple:
        '''function to iterate through each feature and calculate weighted impurity '''

        # values to return from this function as a tuple
        left_subnode_impurity:float = 0
        right_subnode_impurity:float = 0
        weighted_impurity:float = 0


        # outer loop iterate through feature
        for feature in self.feature_names_in_:
            combined:pd.DataFrame = X_segment[feature]
            combined['label'] = y_segment
            # Combine feature column and label to create a df and sort them together in order to decide 
            # threshold for feature
            combined = combined.sort_values(by=feature)
            # inner loop iterate through length of sample for current node to calculate combination of 
            # impurity for left and right subnode and their weight and decide which sample length. to split 
            # it by
            for i in range(1,len(combined)):
                if self.criterion == 'gini':
                    left_temp_impurity = self.gini_index(combined['label'].iloc[0:i])
                    right_temp_impurity = self.gini_index(combined['label'].iloc[i:len(combined)])
                else:
                    left_temp_impurity = self.entropy(combined['label'].iloc[0:i])
                    right_temp_impurity = self.entropy(combined['label'].iloc[i:len(combined)])
                weighted_temp_impurity = (i / len(combined) * left_temp_impurity) + \
                ((len(combined) - i) / len(combined) * right_temp_impurity)
                if weighted_temp_impurity <   


    def partition(self, X_segment:pd.DataFrame, y_segment:pd.Series, depth:int, impurity:None)-> None:
        '''recursive function to partition dataset into decision tree as per criteria (gini or entropy)'''
        # POPULATE ATTRIBUTES
        # ----------------------------------
        # depth
        if depth > self.depth_:
            self.depth_ = depth
        # tree_.n_node_samples
        self.tree_.n_node_samples = np.append(self.tree_.n_node_samples, len(y_segment))
        # tree_.impurity
        if impurity is None:
            if self.criterion == 'gini':
                self.tree_.impurity = np.append(self.tree_.impurity, self.gini_index(y_segment))
            else:
                self.tree_.impurity = np.append(self.tree_.impurity, self.entropy(y_segment))
        else:
            self.tree_.impurity = np.append(self.tree_.impurity, impurity)
        # tree_.value
        value:list = [(y_segment == value).sum().item() for value in self.classes_]
        if len(self.tree_.value) == 0:
            self.tree_.value = np.array([value])
        else:
            self.tree_.value = np.append(self.tree_.value, value, axis=0)

        # CONDITION to recursively partion
        # ----------------------------------
        if len(X_segment) != 1 and len(X_segment) >= self.min_samples_split \
            and (self.max_depth is None or (self.max_depth is not None and depth < self.max_depth)):
            # CALCULATE WHETHER TO SPLIT by checking weight average of criteria of left and right subnode
            #  and compare to current
            # ---------------------------------
            self.check_subnode_split(X_segment, y_segment)

            




    def fit(self, X:pd.DataFrame, y:pd.Series):
        '''fit method'''
        if isinstance(X, pd.DataFrame) is False:
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series) is False:
            y = pd.Series(y)
        self.classes_:np.array = y.unique()
        self.feature_names_in_ = X.columns.values
        print(self.feature_names_in_)
        self.partition(X, y, 0, None)
        print(self.tree_.value)
        return self