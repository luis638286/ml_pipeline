from interfaces.base_preprocessor import BasePreprocessor

class IdentityPreprocessor(BasePreprocessor):
    def fit(self, X):
        return self

    def transform(self, X):
        return X