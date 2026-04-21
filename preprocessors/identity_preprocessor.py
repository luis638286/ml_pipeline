from interfaces.base_preprocessor import BasePreprocessor


class IdentityPreprocessor(BasePreprocessor):
    def fit(self, X):
        """this preprocessor learns nothing"""
        return self

    def transform(self, X):
        """returns the data unchanged """
        return X