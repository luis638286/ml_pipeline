# main.py
import argparse
import random
import numpy as np

from data.loader import load_dataset, chronological_split
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner

# Example baselines (demo only — real experiments come from elsewhere)
from preprocessors.identity_preprocessor import IdentityPreprocessor
from preprocessors.minmax_preprocessor import MinMaxPreprocessor
from models.mean_model import MeanModel
from models.zero_model import ZeroModel


def run_experiments(experiments,
                    repo_id="CitrusBoy/EnergyPriceForecasting",
                    subset="Without_Gas",
                    input_len=168,
                    horizon=48,
                    train_ratio=0.7,
                    val_ratio=0.15,
                    seed=42):
    """Core application entrypoint. Takes a list of experiments, runs them all.

    Experiments can be built elsewhere (by CitrusBoy, by tests, by future code)
    and handed in. The app handles data loading, splitting, windowing, and
    result collection — not which experiments to run.
    """
    random.seed(seed)
    np.random.seed(seed)

    X, y = load_dataset(repo_id, subset)
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(
        X, y, train_ratio, val_ratio
    )

    runner = ExperimentRunner(input_len=input_len, horizon=horizon)
    return runner.run_all(experiments, X_tr, y_tr, X_te, y_te,
                          X_val=X_val, y_val=y_val)


def print_results(results, input_len, horizon):
    header = f"{'experiment':20} | {'window':>10} | {'MAE':>8} | {'RMSE':>8} | {'peak10-MAE':>11}"
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        print(f"{r['experiment_name']:20} | "
              f"{r['input_len']:>4}->{r['horizon']:<3} | "
              f"{m['mae']:>8.3f} | "
              f"{m['rmse']:>8.3f} | "
              f"{m['peak_mae_top10']:>11.3f}")


def demo():
    """Runs the built-in baseline experiments. CitrusBoy will replace or
    extend this list with their own experiments."""
    parser = argparse.ArgumentParser(description="Run baseline experiments.")
    parser.add_argument("--subset", default="Without_Gas")
    parser.add_argument("--input-len", type=int, default=168,
                        help="Lookback window size in hours.")
    parser.add_argument("--horizon", type=int, default=48,
                        help="Forecast horizon in hours.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiments = [
        Experiment("Mean baseline",  IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline",  IdentityPreprocessor(), ZeroModel()),
        Experiment("MinMax + Mean",  MinMaxPreprocessor(),   MeanModel()),
    ]

    results = run_experiments(
        experiments,
        subset=args.subset,
        input_len=args.input_len,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print_results(results, args.input_len, args.horizon)


if __name__ == "__main__":
    demo()