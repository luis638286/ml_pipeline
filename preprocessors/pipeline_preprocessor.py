from interfaces.base_preprocessor import BasePreprocessor


class PipelinePreprocessor(BasePreprocessor):
    def __init__(self, preprocessors):
        if not preprocessors:
            raise ValueError("preprocessors list cannot be empty")

        for preprocessor in preprocessors:
            if not isinstance(preprocessor, BasePreprocessor):
                raise TypeError("All preprocessors must inherit from BasePreprocessor")

        self.preprocessors = preprocessors

    def fit(self, X):
        current_X = X

        for preprocessor in self.preprocessors:
            preprocessor.fit(current_X)
            current_X = preprocessor.transform(current_X)

        return self

    def transform(self, X):
        current_X = X

        for preprocessor in self.preprocessors:
            current_X = preprocessor.transform(current_X)

        return current_X