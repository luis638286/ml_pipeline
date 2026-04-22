import numpy as np
from data.loader import load_dataset, chronological_split
from data.windowing import make_windows
from preprocessors.minmax_preprocessor import MinMaxPreprocessor
from preprocessors.identity_preprocessor import IdentityPreprocessor
from models.mean_model import MeanModel
from models.zero_model import ZeroModel
from pipeline.experiment import Experiment
from pipeline.experiment_runner import ExperimentRunner

INPUT_LEN = 168   # 1 week lookback
HORIZON   = 48    # 48h forecast

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))

def main():
    X, y = load_dataset("final_dataset_full_clean.csv")
    X_tr, y_tr, X_val, y_val, X_te, y_te = chronological_split(X, y)

    # Fit scaler on train, apply to all
    scaler = MinMaxPreprocessor().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    # Window
    Xw_tr, yw_tr = make_windows(X_tr_s, y_tr, INPUT_LEN, HORIZON)
    Xw_te, yw_te = make_windows(X_te_s, y_te, INPUT_LEN, HORIZON)

    experiments = [
        Experiment("Mean baseline", IdentityPreprocessor(), MeanModel()),
        Experiment("Zero baseline", IdentityPreprocessor(), ZeroModel()),
    ]

    runner = ExperimentRunner()
    results = runner.run_all(experiments, Xw_tr, yw_tr, Xw_te)

    for r in results:
        print(f"{r['experiment_name']:20} | MAE: {mae(yw_te, r['predictions']):.3f} €/MWh")