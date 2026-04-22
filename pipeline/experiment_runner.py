class ExperimentRunner:
    def run(self, experiment, X_train, y_train, X_test, epochs: int = 10):
        preprocessor = experiment.preprocessor
        model = experiment.model

        preprocessor.fit(X_train)
        X_train_t = preprocessor.transform(X_train)

        model.fit(X_train_t, y_train, epochs=epochs)

        X_test_t = preprocessor.transform(X_test)
        predictions = model.predict(X_test_t)

        return {
            "experiment_name": experiment.name,
            "predictions": predictions
        }

    def run_all(self, experiments, X_train, y_train, X_test, epochs: int = 10):
        return [self.run(exp, X_train, y_train, X_test, epochs) for exp in experiments]