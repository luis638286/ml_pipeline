from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """train the model on input features X and targets y"""
        pass

    @abstractmethod
    def predict(self, X):
        """return predictions for input features X"""
        pass