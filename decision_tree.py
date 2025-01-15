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
        # array of left and right children index (based on n_node_samples) for prediction purpose when traversing
        # the tree
        self.children_left:np.array = np.array([])
        self.children_right:np.array = np.array([])


class DecisionTreeClassifier(BaseEstimator, TransformerMixin):
    '''class definition'''
    def __init__(self, criterion:str='gini', min_samples_split:int = 2, max_depth:int | None = None):
        self.criterion:str = criterion
        self.min_samples_split:int = min_samples_split
        self.max_depth:int | None = max_depth
        self.classes_:np.array = np.array([])
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
            result = result - ((prob) * math.log2(max(prob, 1e-10)))
        return result
    
    def check_subnode_split(self, combined:pd.DataFrame) -> tuple:
        '''function to iterate through each feature and calculate weighted impurity '''

        # top level values to return from this function as a tuple
        left_subnode_impurity:float = -1
        right_subnode_impurity:float = -1
        weighted_impurity:float = -1
        feature_index:int = -1
        sample_split_index:int = -1
        threshold:float = -1 

        # outer loop iterate through feature
        for feature_index_temp, feature in enumerate(self.feature_names_in_):
            sub_combined:pd.DataFrame = pd.DataFrame(combined[feature], columns=[feature])
            sub_combined['label'] = combined['label']
            # Combine feature column and label to create a df and sort them together in order to decide
            # threshold for feature
            sub_combined = sub_combined.sort_values(by=feature)
            # inner loop iterate through length of sample for current node to calculate combination of
            # impurity for left and right subnode and their weight and decide which sample length to split
            # it by
            for i in range(1,len(sub_combined)):
                if sub_combined[feature].iloc[i] == sub_combined[feature].iloc[i-1]:    
                    continue
                if self.criterion == 'gini':
                    left_temp_impurity = self.gini_index(sub_combined['label'].iloc[0:i])
                    right_temp_impurity = self.gini_index(sub_combined['label'].iloc[i:len(sub_combined)])
                else:
                    left_temp_impurity = self.entropy(sub_combined['label'].iloc[0:i])
                    right_temp_impurity = self.entropy(sub_combined['label'].iloc[i:len(sub_combined)])
                # impurity weighted by samples split at left and right
                weighted_temp_impurity = (i / len(sub_combined) * left_temp_impurity) + \
                ((len(sub_combined) - i) / len(sub_combined) * right_temp_impurity)
                # check calculated impurity is 1) less than at current node and whether less
                # than previous sample split, if yes log values at top level to be returned
                # by function.
                if weighted_temp_impurity < self.tree_.impurity[len(self.tree_.impurity) - 1] \
                    and (weighted_impurity == -1 or (weighted_impurity != -1 and \
                        weighted_temp_impurity < weighted_impurity)):
                    left_subnode_impurity = left_temp_impurity
                    right_subnode_impurity = right_temp_impurity
                    weighted_impurity = weighted_temp_impurity
                    # index of feature
                    feature_index = feature_index_temp
                    # index of split
                    sample_split_index = i
                    # calculate mid-point value of split of samples for threshold
                    threshold = (float(sub_combined[feature].iloc[i - 1]) + float(sub_combined[feature].iloc[i])) / 2
        return (left_subnode_impurity, right_subnode_impurity, feature_index, sample_split_index, threshold)
                    
                    


    def partition(self, combined:pd.DataFrame, depth:int, impurity:float | None)-> None:
        '''recursive function to partition dataset into decision tree as per criteria (gini or entropy)'''
        # POPULATE ATTRIBUTES
        # ----------------------------------
        # depth
        if depth > self.depth_:
            self.depth_ = depth
        # tree_.n_node_samples
        self.tree_.n_node_samples = np.append(self.tree_.n_node_samples, len(combined))
        # tree_.impurity
        # for root node, calculate impurity
        if impurity is None:
            if self.criterion == 'gini':
                self.tree_.impurity = np.append(self.tree_.impurity, self.gini_index(combined['label']))
            else:
                self.tree_.impurity = np.append(self.tree_.impurity, self.entropy(combined['label']))
        else:
            self.tree_.impurity = np.append(self.tree_.impurity, impurity)
        # tree_.value
        value:list = [(combined['label'] == value).sum().item() for value in self.classes_]
        if len(self.tree_.value) == 0:
            self.tree_.value = np.array([value])
        else:
            self.tree_.value = np.append(self.tree_.value, [value], axis=0)
        # check depth, if > instance's depth_, replace value
        if depth > self.depth_:
            self.depth_ = depth

        # HYPERPARAM CONDITIONs to check whether to recursively partition
        # ---------------------------------------------------------------
        if self.tree_.n_node_samples[len(self.tree_.n_node_samples) - 1] > 1 and \
            self.tree_.n_node_samples[len(self.tree_.n_node_samples) - 1] >= self.min_samples_split \
            and (self.max_depth is None or (self.max_depth is not None and depth < self.max_depth)):
            # CALCULATE whether to split and values of the node
            # --------------------------------------------------
            left_subnode_impurity, right_subnode_impurity, feature_index, sample_split_index, threshold = \
                self.check_subnode_split(combined)
            # append these 2 values at index of location of node regardless whether split or not
            self.tree_.feature = np.append(self.tree_.feature, feature_index)
            self.tree_.threshold = np.append(self.tree_.threshold, threshold)
            if left_subnode_impurity != -1:
                # sort based on feature return by check_subnode_split then only split by sample index
                combined = combined.sort_values(by=combined.columns.values[feature_index])
                # split left node
                self.tree_.children_left = np.append(self.tree_.children_left, len(self.tree_.n_node_samples))
                self.tree_.children_right = np.append(self.tree_.children_right, -1)
                self.partition(combined[0:sample_split_index], depth + 1, left_subnode_impurity)
                # split right node
                self.tree_.children_right = np.append(len(self.tree_.n_node_samples),self.tree_.children_right)
                self.tree_.children_left = np.append(self.tree_.children_left, -1)
                self.partition(combined[sample_split_index:len(combined)], depth + 1, right_subnode_impurity)

        # if don't partition as per hyperparams, append -1 in relevant values at index of location of node
        # to indicate that they are leaf
        else:
            self.tree_.feature = np.append(self.tree_.feature, -1)
            self.tree_.threshold = np.append(self.tree_.threshold, -1)

    def fit(self, X:pd.DataFrame, y:pd.Series):
        '''fit method'''
        # accepts numpy params as well. Convert to DF
        if isinstance(X, pd.DataFrame) is False:
            X = pd.DataFrame(X)
        if isinstance(y, pd.Series) is False:
            y = pd.Series(y)
        # array of classes value from y
        self.classes_:np.array = y.unique()
        # array of feature names
        self.feature_names_in_ = X.columns.values
        # combine to one df for ease of sorting
        combined:pd.DataFrame = X
        combined['label'] = y
        self.partition(combined, 0, None)
        print("node",self.tree_.n_node_samples)
        print("threshold",self.tree_.threshold)
        print("impurity",self.tree_.impurity)
        print("feature",self.tree_.feature)
        print("children_left", self.tree_.children_left)
        print("children_right", self.tree_.children_right)
        # print("value",self.tree_.value)
        return self

    def eval_node(self, index:int) -> int:
        '''eval leaf node and return predict node based on self.tree_.value at node'''
        return np.argmax(self.tree_.value[index])




    def predict(self, X:pd.DataFrame)->np.array:
        '''predict method'''
        if isinstance(X, pd.DataFrame) is False:
            X = pd.DataFrame(X)
        prediction:np.array = np.array([])
        for _, row in X.iterrows():
            j = 0
            while j < len(self.tree_.n_node_samples):
                if self.tree_.feature[j] == -1:
                    prediction = np.append(prediction, self.classes_[self.eval_node(j)])
                    break
                elif row.iloc[int(self.tree_.feature[j])] <= self.tree_.threshold[j]:
                    j = j + 1
                elif row.iloc[int(self.tree_.feature[j])] > self.tree_.threshold[j]:
                    for k in range(j + 2, len(self.tree_.n_node_samples)):
                        if self.tree_.n_node_samples[j] - self.tree_.n_node_samples[j + 1] == self.tree_.n_node_samples[k]:
                            j = k
                            break
        return prediction
                        
                    



        
        
