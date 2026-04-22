from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Train the model. kwargs for model-specific params (epochs, lr, etc)"""
        pass

    @abstractmethod
    def predict(self, X):
        pass