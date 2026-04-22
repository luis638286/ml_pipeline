import pandas as pd

def load_dataset(path, target_col="price"):
    df = pd.read_csv(path, parse_dates=["time"]).sort_values("time")
    df = df.set_index("time")
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    return X, y

def chronological_split(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (X[:train_end], y[:train_end],
            X[train_end:val_end], y[train_end:val_end],
            X[val_end:], y[val_end:])