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


ID = f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
FOLDER = "pipelines/results"
OUTPUT = "pipelines/output"
PDFOUTPUT = f"{OUTPUT}/{ID}.pdf"
TXTOUTPUT = f"{OUTPUT}/{ID}.txt"
# Filenames structure: {db}-{model}-{vm}-{trainw}-{testw}.csv


def splitname(filename):
    last = filename.split(f"results/")[-1].split(".")[0]
    return last.split("-")


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


def getpredictions(filename):
    filename = f"pipelines/results/{filename}"
    df = pd.read_csv(filename)

    return df


def getmetrics(ytrue, yhat):
    r2 = r2_score(ytrue, yhat)
    mae = mean_absolute_error(ytrue, yhat)
    mse = mean_squared_error(ytrue, yhat, squared=True)
    rmse = mean_squared_error(ytrue, yhat, squared=False)

    return r2, mae, mse, rmse


def scatter(df, pdf):
    for index, row in df.iterrows():
        fig, ax = plt.subplots()

        ax.plot(row.ytrue, row.yhat, "o", color="tab:blue")
        ax.set_xlabel("ytrue")
        ax.set_ylabel("yhat")
        ax.set_title(index)

        textstr = (
            f"R2:   {round(row.r2, 2)}\n"
            f"MAE:  {round(row.mae, 2)}\n"
            f"MSE:  {round(row.mse, 2)}\n"
            f"RMSE: {round(row.rmse, 2)}"
        )

        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.025,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        plot.wrapup(pdf)


def scatter4subplot(df, pdf):
    n = len(df.index)
    for i in range(n):
        if i % 4 != 0:
            continue

        fig, ax = plt.subplots(2, 2)
        ax = ax.flatten()

        for j, k in enumerate(range(i, i + 4)):
            idx = df.index[k]
            row = df.iloc[k]

            x = row.ytrue
            y = row.yhat

            # Calculate the point density
            xy = np.vstack([x, y])
            z = scipy.stats.gaussian_kde(xy)(xy)

            # Sort the points by density, so that the densest points are plotted last
            # idx = z.argsort()
            # x, y, z = x[idx], y[idx], z[idx]

            # ax[j].plot(x, y, "o", color="tab:blue")
            ax[j].scatter(x, y, c=z, s=50)
            ax[j].set_xlabel("ytrue")
            ax[j].set_ylabel("yhat")
            ax[j].set_title(idx)

            # Plot 'perfect' line
            ax[j].plot(x, x, "--", color="tab:gray")

            textstr = (
                f"R2:   {round(row.r2, 3)}\n"
                f"MAE:  {round(row.mae, 3)}\n"
                f"MSE:  {round(row.mse, 3)}\n"
                f"RMSE: {round(row.rmse, 3)}"
            )

            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax[j].text(
                0.025,
                0.95,
                textstr,
                transform=ax[j].transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )

        plot.wrapup(pdf)


def barofbests(df, pdf):
    lr = df.filter(like="LR", axis="index")
    lr = lr[lr.rmse == lr.rmse.min()]

    knn = df.filter(like="KNN", axis="index")
    knn = knn[knn.rmse == knn.rmse.min()]

    svr = df.filter(like="SVR", axis="index")
    svr = svr[svr.rmse == svr.rmse.min()]

    df = pd.concat([lr, knn, svr])
    df = df.drop(["ytrue", "yhat"], axis=1)

    df = df.drop("r2", axis=1)

    df.plot.bar(rot=0, grid=True, secondary_y="mse")
    plt.title("Best model of each method")
    plot.wrapup(pdf)


def lineofbests(df, pdf):
    methods = ["LR", "KNN", "SVR"]
    tmp = []

    for method in methods:
        filtered = df.filter(like=method, axis="index")
        filtered = filtered[filtered.rmse == filtered.rmse.min()]

        if filtered.empty:
            continue

        filename = filtered.iloc[0].name + ".csv"

        dfpred = getpredictions(filename)
        tmp.append(dfpred)

        dfpred.plot(rot=0, grid=True)
        plt.title(f"Best {method} method: {filename}")

        # ==============================================================================
        abserror = abs(dfpred.ytrue - dfpred.yhat)
        idx = np.argsort(abserror)

        best = idx[:25]
        worst = np.flip(idx)[:25]

        # TODO: explore worst AND best dataframes
        #  Is it usually in a specific weekday?
        #  Is there any pattern?
        #  Do it in a modularized scrip?
        #  Analise metrics of these dataframes

        plt.plot(
            worst, dfpred.yhat[worst], linestyle="None", marker="o", color="tab:red"
        )
        plt.plot(
            best, dfpred.yhat[best], linestyle="None", marker="o", color="tab:green"
        )
        # ==============================================================================

        plot.wrapup(pdf)

        # modelname = filtered.iloc[0].name
        # data = {"ytrue": filtered.ytrue, "yhat": filtered.yhat.to_numpy()}
        # filtered = pd.DataFrame(data=data)
        #
        # filtered.plot(rot=0, grid=True)
        # plt.title(f"Best {method} method: {modelname}")
        # plot.wrapup(pdf)

    # df = pd.concat(tmp)
    # df.plot(rot=0, grid=True)
    # plt.title(f"Best model of each method")
    # plot.wrapup(pdf)


def table_(df):
    df = df.sort_values(by=["rmse"])

    print(f"Index               RMSE    R2")
    print("==================================")
    for index, row in df.iterrows():
        print(f"{index} \t{round(row.rmse, 3)} \t{round(row.r2, 3)}")


def tablebyfs(df):
    fslist = ["FS1", "FS2", "FS3", "FS4"]
    for fs in fslist:
        filtered = df.filter(like=fs, axis="index")
        filtered = filtered.sort_values(by=["rmse"])

        print(f"Index               RMSE    R2")
        print("==================================")
        for index, row in filtered.iterrows():
            print(f"{index} \t{round(row.rmse, 3)} \t{round(row.r2, 3)}")

        if fs != fslist[-1]:
            print("----------------------------------")


def tablebymethod(df):
    methods = ["LR", "KNN", "SVR"]
    for method in methods:
        filtered = df.filter(like=method, axis="index")
        filtered = filtered.sort_values(by=["rmse"])

        print(f"Index               RMSE    R2")
        print("==================================")
        for index, row in filtered.iterrows():
            print(f"{index} \t{round(row.rmse, 3)} \t{round(row.r2, 3)}")

        if method != methods[-1]:
            print("----------------------------------")


def table(df, methods, fslist):
    length = 54

    print("=" * length)
    print(f"Method\tFS1\t\t\tFS2\t\t\tFS3\t\t\tFS4")
    print("\t\t" + "-" * (length - 8))
    print(f"\t\tRMSE  R2    RMSE  R2    RMSE  R2    RMSE  R2")
    print("\t\t----------\t----------\t----------\t----------")

    for method in methods:
        print(f"{method[:6]}\t\t", end="")

        for fs in fslist:
            like = f"{fs}-{method}"
            filtered = df.filter(like=like, axis="index")
            filtered = filtered.sort_values(by=["rmse"])

            if filtered.empty:
                continue

            best = filtered.iloc[0]
            print(f"{round(best.rmse, 2):.2f}  {round(best.r2, 2):.2f}\t", end="")

        print()

    print("=" * length)


def main():
    if not os.path.isdir(OUTPUT):
        os.makedirs(OUTPUT, exist_ok=True)

    sys.stdout = open(TXTOUTPUT, "w")

    methods = ["LR", "KNN", "SVR"]
    fslist = ["FS1", "FS2", "FS3", "FS4"]

    df = getresults()

    table(df, methods, fslist)
    with matplotlib.backends.backend_pdf.PdfPages(PDFOUTPUT) as pdf:
        scatter4subplot(df, pdf)
        barofbests(df, pdf)
        lineofbests(df, pdf)

    print()
    tablebyfs(df)
    print()
    tablebymethod(df)

    sys.stdout.close()


if __name__ == "__main__":
    main()

# ======================================================================================
# "Estudo de métodos de aprendizado de máquina para estimação da demanda de água em uma
# cidade turística litorânea"
# "Study of machine learning methods to estimate water demand in a coastal tourist town
#
# X configurações dos métodos (e.g., 1 LR, 5 KNN, ...)
#
# Aplicados em cada FS (FS1, FS2, FS3 e FS4)
# Aplicados com cada configuração de janela (EW and SW, with different sizes)
# ======================================================================================
