from interfaces.base_preprocessor import BasePreprocessor
import numpy as np
class MinMaxPreprocessor(BasePreprocessor):
    def fit(self, X):
        X = np.asarray(X)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        self.span = np.where(self.maxs - self.mins == 0, 1, self.maxs - self.mins)
        return self

    def transform(self, X):
        return (np.asarray(X) - self.mins) / self.span