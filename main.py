from preprocessors.identity_preprocessor import IdentityPreprocessor
from preprocessors.minmax_preprocessor import MinMaxPreprocessor
from preprocessors.pipeline_preprocessor import PipelinePreprocessor
from models.dummy_model import DummyModel
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner
import random

def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)

def generate_dataset(n_samples=100):
    X = []
    y = []

    for i in range(n_samples):
        # Features
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)

        # Add some pattern + noise
        noise = random.uniform(-1, 1)

        # Simple rule: if sum > threshold → class 1
        label = 1 if (x1 + x2 + noise) > 10 else 0

        X.append([x1, x2])
        y.append(label)

    return X, y


def main():
    # Generate data
    X, y = generate_dataset(200)

    # Split manually (80/20)
    split_index = int(0.8 * len(X))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    # Single preprocessor
    preprocessor1 = MinMaxPreprocessor()

    # Another preprocessor
    preprocessor2 = IdentityPreprocessor()

    # Chain them into one pipeline
    combined_preprocessor = PipelinePreprocessor([
        preprocessor1,
        preprocessor2
    ])

    model = DummyModel()

    experiment = Experiment(
        name="Exp.1.a - MinMax + Identity + DummyModel",
        preprocessor=combined_preprocessor,
        model=model
    )
    
    runner = ExperimentRunner()
    result = runner.run(experiment, X_train, y_train, X_test)
    predictions = result["predictions"]

    acc = accuracy(y_test, predictions)

    print("Experiment:", result["experiment_name"])
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()