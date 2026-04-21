class ExperimentRunner:
    def run(self, experiment, X_train, y_train, X_test):
        preprocessor = experiment.preprocessor
        model = experiment.model

        preprocessor.fit(X_train)
        X_train_transformed = preprocessor.transform(X_train)

        model.fit(X_train_transformed, y_train)

        X_test_transformed = preprocessor.transform(X_test)
        predictions = model.predict(X_test_transformed)

        return {
            "experiment_name": experiment.name,
            "predictions": predictions
        }