import numpy as np
from data.loader import load_dataset, chronological_split
from data.windowing import make_windows
from preprocessors.minmax_preprocessor import MinMaxPreprocessor
from preprocessors.identity_preprocessor import IdentityPreprocessor
from models.mean_model import MeanModel
from models.zero_model import ZeroModel
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner

# Defaults — adjustable via main() arguments
HF_REPO_DEFAULT     = "CitrusBoy/EnergyPriceForecasting"
HF_SUBSET_DEFAULT   = "Without_Gas"
INPUT_LEN_DEFAULT   = 168    # 1 week lookback
HORIZON_DEFAULT     = 48     # 48h forecast
TRAIN_RATIO_DEFAULT = 0.7
VAL_RATIO_DEFAULT   = 0.15

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

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
        Experiment("Mean (2wk->1d)",  IdentityPreprocessor(), MeanModel(),
                   input_len=336, horizon=24),
    ]

    runner = ExperimentRunner()
    for exp in experiments:
        il = exp.input_len if exp.input_len is not None else input_len
        hz = exp.horizon   if exp.horizon   is not None else horizon

        Xw_tr, yw_tr = make_windows(X_tr, y_tr, il, hz)
        Xw_te, yw_te = make_windows(X_te, y_te, il, hz)

        result = runner.run(exp, Xw_tr, yw_tr, Xw_te)
        print(f"{result['experiment_name']:20} | "
              f"win={il}->{hz} | "
              f"MAE: {mae(yw_te, result['predictions']):.3f} €/MWh")

if __name__ == "__main__":
    main()
