from interfaces.base_model import BaseModel
import numpy as np

class ZeroModel(BaseModel):
    """Predicts 0 for every input"""
    def fit(self, X, y, **kwargs):
        self.horizon = y.shape[1] if y.ndim > 1 else 1
        return self

    def predict(self, X):
        return np.zeros((len(X), self.horizon))
