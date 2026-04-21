from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    @abstractmethod
    def fit(self, X):
        """learn anything needed from the training data"""
        pass

    @abstractmethod
    def transform(self, X):
        """transform the input data and return the transformed version"""
        pass