class ExperimentRunner:
    def run(self, experiment, X_train, y_train, X_test, **fit_kwargs):
        experiment.preprocessor.fit(X_train)
        X_train_t = experiment.preprocessor.transform(X_train)

        experiment.model.fit(X_train_t, y_train, **fit_kwargs)

        X_test_t = experiment.preprocessor.transform(X_test)
        predictions = experiment.model.predict(X_test_t)

        return {
            "experiment_name": experiment.name,
            "predictions": predictions,
        }

    def run_all(self, experiments, X_train, y_train, X_test, **fit_kwargs):
        return [self.run(exp, X_train, y_train, X_test, **fit_kwargs)
                for exp in experiments]