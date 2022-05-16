import os

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import src.plot as plot

# ======================================================================================
# FIXME: worst and best are not of type datetime when calling dfpred.index[]
# abserror = abs(dfpred.ytrue - dfpred.yhat)
# idx = np.argsort(abserror)
#
# n = 10
# best = idx[:n].to_list()
# worst = np.flip(idx)[:n].to_list()
#
# plt.plot(
#     dfpred.index[worst], dfpred.yhat[worst], linestyle="None", marker="o",
#     color="tab:red"
# )
# plt.plot(
#     dfpred.index[best], dfpred.yhat[best], linestyle="None", marker="o",
#     color="tab:green"
# )
# ======================================================================================


def _getmetrics(ytrue, yhat, decimals=2):
    r2 = round(r2_score(ytrue, yhat), decimals)
    mae = round(mean_absolute_error(ytrue, yhat), decimals)
    mape = round(mean_absolute_percentage_error(ytrue, yhat), decimals)
    rmse = round(mean_squared_error(ytrue, yhat, squared=False), decimals)

    return r2, mae, mape, rmse


class Sink:
    def __init__(self, prs, path):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        self.prs = prs
        self.output = path
        self.metric = "rmse"
        self.methods = ["LR", "KNN", "SVR", "MLP"]

        self.metrics = self._prs2df()
        dst = os.path.join(self.output, "metrics.csv")
        self.metrics.to_csv(dst, index_label="model")

    def _prs2df(self):
        index = []
        data = dict(r2=[], mae=[], mape=[], rmse=[], split=[])
        for split in ["val", "test"]:
            for pr in self.prs:
                df = pr.df[pr.df.split == split]

                r2, mae, mape, rmse = _getmetrics(df.ytrue, df.yhat)

                index.append(pr.id)
                data["r2"].append(r2)
                data["mae"].append(mae)
                data["mape"].append(mape)
                data["rmse"].append(rmse)
                data["split"].append(split)

        # Create dataframe
        df = pd.DataFrame(data=data, index=index)

        return df

    def _getbest(self, like, split="test"):
        subset = self.metrics[self.metrics.split == split]
        subset = subset.filter(like=like, axis="index")

        if self.metric != "r2":
            best = subset[subset[self.metric] == subset[self.metric].min()]
        else:
            best = subset[subset[self.metric] == subset[self.metric].max()]

        return best

    def _getbests(self):
        # TODO: if more than one, select by validation
        lr = self._getbest("LR").head(1)
        knn = self._getbest("KNN")
        svr = self._getbest("SVR")
        mlp = self._getbest("MLP")

        return pd.concat([lr, knn, svr, mlp])

    def _getpredictions(self, id):
        for pr in self.prs:
            if pr.id == id:
                data = {"yhat": pr.yhat, "ytrue": pr.ytrue}
                df = pd.DataFrame(data=data, index=pr.timestamp)
                df.index = pd.to_datetime(df.index)
                return df

    def evaluate(self, overview=True, detailed=True):
        output = f"{self.output}/output.pdf"
        with matplotlib.backends.backend_pdf.PdfPages(output) as pdf:
            if overview:
                self.barofbests(pdf)
                self.lineofbests(pdf)

            if detailed:
                pass
                # table()
                # scatter4subplot()

            # Temporary
            self.weekly(pdf)
            self.monthly(pdf)

    def barofbests(self, pdf):
        df = self._getbests()
        title = f"Model with min {self.metric.upper()} of each method"

        axes = df.plot.bar(rot=0, grid=True, subplots=True, layout=(2, 2), title=title)
        for ax in axes.flatten():
            for container in ax.containers:
                ax.bar_label(container)

        plot.wrapup(pdf)

    def lineofbests(self, pdf):
        for method in self.methods:
            # Select the model with best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            dfpred = self._getpredictions(filtered.index[0])
            ls = ["--", "-"]
            ax = dfpred.plot(rot=0, grid=True, style=ls)

            # markers = ["v", "o"]
            # for i, line in enumerate(ax.get_lines()):
            #     line.set_marker(markers[i])

            plt.title(f"Best {method} method: {filtered.index[0]}")
            plot.wrapup(pdf)

    def weekly(self, pdf):
        for method in self.methods:
            filtered = self._getbest(method)
            if filtered.empty:
                continue

            predictions = self._getpredictions(filtered.index[0])

            mondays = np.where(predictions.index.weekday == 0)[0]
            mondays = np.random.choice(mondays, 4)
            fig, axes = plt.subplots(2, 2)

            for ax, startrow in zip(axes.flatten(), mondays):
                endrow = startrow + 7
                chunk = predictions.iloc[startrow:endrow, :]

                ls = ["--", "-"]
                title = f"{chunk.index[0].date()} to {chunk.index[-1].date()}"
                chunk.index = chunk.index.day_name()
                chunk.plot(rot=0, grid=True, style=ls, ax=ax, title=title)

            markers = ["v", "o"]
            for i, ax in enumerate(axes.flatten()):
                if i < 2:
                    ax.set_xticklabels([])

                # for j, line in enumerate(ax.get_lines()):
                #     line.set_marker(markers[j])

            fig.suptitle(f"Random weekly predictions")
            plot.wrapup(pdf)

    def monthly(self, pdf):
        for method in self.methods:
            filtered = self._getbest(method)
            if filtered.empty:
                continue

            predictions = self._getpredictions(filtered.index[0])

            firstday = np.where(predictions.index.day == 1)[0]
            firstday = np.random.choice(firstday, 4)
            fig, axes = plt.subplots(2, 2)

            for ax, startrow in zip(axes.flatten(), firstday):
                endrow = startrow + 30
                chunk = predictions.iloc[startrow:endrow, :]

                ls = ["--", "-"]
                title = f"{chunk.index[0].strftime('%b')}"
                chunk.index = chunk.index.day_name().str.slice(0, 3)
                chunk.plot(rot=90, grid=True, style=ls, ax=ax, title=title)

                ax.set_xticks(np.arange(len(chunk.index)), chunk.index, rotation=90)

            # markers = ["v", "o"]
            # for i, ax in enumerate(axes.flatten()):
            #     # if i < 2:
            #     #     ax.set_xticklabels([])
            #
            #     for j, line in enumerate(ax.get_lines()):
            #         line.set_marker(markers[j])

            fig.suptitle(f"Random monthly predictions")
            plot.wrapup(pdf)
