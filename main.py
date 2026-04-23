from data.loader import load_dataset, chronological_split
from preprocessors.minmax_preprocessor import MinMaxPreprocessor
from preprocessors.identity_preprocessor import IdentityPreprocessor
from models.mean_model import MeanModel
from models.zero_model import ZeroModel
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner

HF_REPO_DEFAULT     = "CitrusBoy/EnergyPriceForecasting"
HF_SUBSET_DEFAULT   = "Without_Gas"
INPUT_LEN_DEFAULT   = 168    # 1 week lookback
HORIZON_DEFAULT     = 48     # 48h forecast
TRAIN_RATIO_DEFAULT = 0.7
VAL_RATIO_DEFAULT   = 0.15


def main(repo_id=HF_REPO_DEFAULT,
         subset=HF_SUBSET_DEFAULT,
         input_len=INPUT_LEN_DEFAULT,
         horizon=HORIZON_DEFAULT,
         train_ratio=TRAIN_RATIO_DEFAULT,
         val_ratio=VAL_RATIO_DEFAULT):
    X, y = load_dataset(repo_id, subset)
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(
        X, y, train_ratio, val_ratio
    )

    experiments = [
        Experiment("Mean baseline",   IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline",   IdentityPreprocessor(), ZeroModel()),
        Experiment("MinMax + Mean",   MinMaxPreprocessor(),   MeanModel()),
        Experiment("Mean (2wk->1d)",  IdentityPreprocessor(), MeanModel(), input_len=336, horizon=24),
    ]

    runner = ExperimentRunner(input_len=input_len, horizon=horizon)
    results = runner.run_all(experiments, X_tr, y_tr, X_te, y_te, X_val=X_val, y_val=y_val)

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


if __name__ == "__main__":
    main()