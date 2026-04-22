from interfaces.base_model import BaseModel
import numpy as np

class MeanModel(BaseModel):
    """Predicts per-horizon mean of training targets."""
    def fit(self, X, y, **kwargs):
        self.mean = np.mean(y, axis=0)  # shape: (horizon,)
        return self

    def predict(self, X):
        return np.tile(self.mean, (len(X), 1))  # (n_samples, horizon)