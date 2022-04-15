import glob
import os

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pipelines.utils as pipeutils


DBFILE = "data/dataset.csv"
FOLDER = "pipelines/results"


def getresults():
    filelist = glob.glob(os.path.join(FOLDER, "*.csv"))
    idx, r2, mae, mse, rmse, ytrue, yhat = [], [], [], [], [], [], []

    # Read files and store values in lists
    for file in sorted(filelist):
        df = pd.read_csv(file)

        _idx = file.split(f"results/")[-1].split(".")[0]
        _ytrue, _yhat = df.ytrue, df.yhat
        _r2, _mae, _mse, _rmse = getmetrics(df.ytrue, df.yhat)

        idx.append(_idx)
        r2.append(_r2)
        mae.append(_mae)
        mse.append(_mse)
        rmse.append(_rmse)
        ytrue.append(_ytrue)
        yhat.append(_yhat)

    # Create dataframe
    data = {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "ytrue": ytrue,
        "yhat": yhat,
    }
    df = pd.DataFrame(data=data, index=idx)

    return df


def getmetrics(ytrue, yhat):
    r2 = r2_score(ytrue, yhat)
    mae = mean_absolute_error(ytrue, yhat)
    mse = mean_squared_error(ytrue, yhat, squared=True)
    rmse = mean_squared_error(ytrue, yhat, squared=False)

    return r2, mae, mse, rmse


def getfs():
    fs = pd.read_csv(DBFILE)
    # fs = pipeutils.getsubset(fs, "FS1")
    fs = fs.dropna()

    return fs


def getpredictions(filename):
    filename = f"pipelines/results/{filename}"
    return pd.read_csv(filename)


def getpred():
    forecast = getresults()
    forecast = forecast.filter(like="FS1", axis="index")
    forecast = forecast.sort_values(by=["rmse"])

    id = forecast.index[0]
    bestmetrics = forecast.iloc[0]
    bestpred = getpredictions(f"{id}.csv")

    return bestpred


def getbestidx(df):
    abserror = abs(df.ytrue - df.yhat)
    return np.argsort(abserror)


def main():
    fs = getfs()
    pred = getpred()

    best = getbestidx(pred) + 730
    fsordered = fs.iloc[best]

    full = fsordered.describe()
    head = fsordered.head(30).describe()
    tail = fsordered.tail(30).describe()

    print()


if __name__ == "__main__":
    main()
