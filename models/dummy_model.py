from interfaces.base_model import BaseModel


class DummyModel(BaseModel):
    def __init__(self):
        self.most_common_class = None

    def fit(self, X, y):
        """store the most common class from y"""
        counts = {}

        for label in y:
            counts[label] = counts.get(label, 0) + 1

        self.most_common_class = max(counts, key=counts.get)
        return self

    def predict(self, X):
        """predict the most common class for every row in X"""
        if self.most_common_class is None:
            raise ValueError("Model must be fitted before calling predict()")

        return [self.most_common_class for _ in X]