from preprocessors.minmax_preprocessor import MinMaxPreprocessor
from preprocessors.identity_preprocessor import IdentityPreprocessor
from models.dummy_model import DummyModel
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner
import random

def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

def generate_dataset(n=200):
    X, y = [], []
    for _ in range(n):
        x1, x2 = random.uniform(0, 10), random.uniform(0, 10)
        y.append(1 if (x1 + x2 + random.uniform(-1, 1)) > 10 else 0)
        X.append([x1, x2])
    return X, y

def main():
    X, y = generate_dataset()
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    experiments = [
        Experiment("Exp.1a - MinMax + Dummy",    MinMaxPreprocessor(), DummyModel()),
        Experiment("Exp.1b - Identity + Dummy",  IdentityPreprocessor(), DummyModel()),
    ]

    runner = ExperimentRunner()
    results = runner.run_all(experiments, X_train, y_train, X_test)

    for result in results:
        acc = accuracy(y_test, result["predictions"])
        print(f"{result['experiment_name']} | Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()