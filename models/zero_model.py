from interfaces.base_model import BaseModel

class ZeroModel(BaseModel):
    """Predicts 0 for every input"""
    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X):
        return [0.0 for _ in X]