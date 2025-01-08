'''Own Principal Component Analysis implementation using numpy'''

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class PCA(BaseEstimator, TransformerMixin):
    '''class definition for PCA with fit and transform function inheriting from BaseEstimator with defined functions
    (for getting params used in GridSearchCV and friends) and TransformerMixin(Used by pipeline) which has
    fit_transform method which delegates to fit AND transform method which this class has to define (abstract_methods in
    TransformerMixin)'''
    def __init__(self, n_components:int=None):
        '''Params for initialization, only use one. No random_state as np.linalg.svd is fully deterministic and not
        sample based projections'''
        self.n_components:int = n_components
        self.components_:np.ndarray | None = None
 
    def fit(self, X:np.ndarray, y=None):
        '''fit'''
        if self.n_components is None:
            self.n_components = X.shape[1]
        elif self.n_components > X.shape[1]:
            raise ValueError("n_components cannot be greater than the number of features in X.")
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        return self

    def transform(self, X:np.ndarray):
        '''transform'''
        return np.dot(X, self.components_.T)
