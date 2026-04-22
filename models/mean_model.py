from interfaces.base_model import BaseModel

class MeanModel(BaseModel):
    """Predicts the mean of training targets for every input"""
    def fit(self, X, y, **kwargs):
        self.mean = sum(y) / len(y)
        return self

    def predict(self, X):
        return [self.mean for _ in X]