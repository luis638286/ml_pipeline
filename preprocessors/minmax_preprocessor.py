from interfaces.base_preprocessor import BasePreprocessor

class MinMaxPreprocessor(BasePreprocessor):
    def fit(self, X):
        n_features = len(X[0])
        self.mins = [min(row[i] for row in X) for i in range(n_features)]
        self.maxs = [max(row[i] for row in X) for i in range(n_features)]
        return self

    def transform(self, X):
        result = []
        for row in X:
            new_row = []
            for i, val in enumerate(row):
                span = self.maxs[i] - self.mins[i]
                new_row.append(0.0 if span == 0 else (val - self.mins[i]) / span)
            result.append(new_row)
        return result