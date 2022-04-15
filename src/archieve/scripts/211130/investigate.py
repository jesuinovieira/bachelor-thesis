import datetime
import glob
import os
import sys

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import src.lib.plot as plot


FOLDER = "pipelines/results"
PDF = "output/output.pdf"
DBFILE = "data/dataset.csv"


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


def getpredictions(filename):
    filename = f"pipelines/results/{filename}"
    return pd.read_csv(filename)


def xxx(fs, pred):
    methods = ["LR", "KNN", "SVR"]
    tmp = []

    for method in methods:
        filtered = pred.filter(like=method, axis="index")
        filtered = filtered[filtered.rmse == filtered.rmse.min()]

        if filtered.empty:
            continue

        filename = filtered.iloc[0].name + ".csv"

        dfpred = getpredictions(filename)
        tmp.append(dfpred)

        dfpred.plot(rot=0, grid=True)
        plt.title(f"Best {method} method: {filename}")

        abserror = abs(dfpred.ytrue - dfpred.yhat)
        idx = np.argsort(abserror)

        # best = dfpred.iloc[idx]
        # worst = dfpred.iloc[np.flip(idx)]

        print()


def main():
    with matplotlib.backends.backend_pdf.PdfPages(PDF) as pdf:
        import pipelines.utils as pipeutils

        fs = pd.read_csv(DBFILE)
        fs = pipeutils.getsubset(fs, "FS1")
        fs = fs.dropna()

        pred = getresults()
        xxx(fs, pred)


if __name__ == "__main__":
    main()
