from interfaces.base_preprocessor import BasePreprocessor


class MinMaxPreprocessor(BasePreprocessor):
    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit(self, X):
        if not X:
            raise ValueError("X cannot be empty.")

        num_features = len(X[0])
        self.mins = [float("inf")] * num_features
        self.maxs = [float("-inf")] * num_features

        for row in X:
            if len(row) != num_features:
                raise ValueError("All rows in X must have the same number of features.")

            for i, value in enumerate(row):
                if value < self.mins[i]:
                    self.mins[i] = value
                if value > self.maxs[i]:
                    self.maxs[i] = value

        return self

    def transform(self, X):
        if self.mins is None or self.maxs is None:
            raise ValueError("Preprocessor must be fitted before calling transform().")

        transformed = []

        for row in X:
            if len(row) != len(self.mins):
                raise ValueError("Input row has a different number of features than the fitted data.")

            new_row = []
            for i, value in enumerate(row):
                min_val = self.mins[i]
                max_val = self.maxs[i]

                if max_val == min_val:
                    new_row.append(0.0)
                else:
                    scaled_value = (value - min_val) / (max_val - min_val)
                    new_row.append(scaled_value)

            transformed.append(new_row)

        return transformed