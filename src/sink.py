import os

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

linestyles = (
    ('solid', (0, ())),
    # ('loosely dotted', (0, (1, 10))),
    # ('dotted', (0, (1, 5))),
    # ('densely dotted', (0, (1, 1))),

    # ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),

    # ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),

    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
)


def _getmetrics(ytrue, yhat, decimals=2):
    r2 = round(r2_score(ytrue, yhat), decimals)
    mae = round(mean_absolute_error(ytrue, yhat), decimals)
    mape = round(mean_absolute_percentage_error(ytrue, yhat), decimals)
    rmse = round(mean_squared_error(ytrue, yhat, squared=False), decimals)

    return r2, mae, mape, rmse


def _bestidx(df, method, metric):
    df = df.filter(like=method, axis="index")

    condition = df[metric].min() if metric != "r2" else df[metric].max()
    df = df[df[metric] == condition]

    return df.index.to_list()


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

        sns.set_theme()
        sns.set_style("whitegrid")

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

    def _getbest(self, like):
        subset = self.metrics[self.metrics.split == "test"]
        subset = subset.filter(like=like, axis="index")

        if self.metric != "r2":
            best = subset[subset[self.metric] == subset[self.metric].min()]
        else:
            best = subset[subset[self.metric] == subset[self.metric].max()]

        # TODO: if more than one, select by validation
        if len(best) > 1:
            idx = best.index[0]

        return best

    def _boem(self):
        # Best of each method
        testmetrics = self.metrics[self.metrics.split == "test"]
        valmetrics = self.metrics[self.metrics.split == "val"]

        indexes = []
        for method in self.methods:
            index = _bestidx(testmetrics, method, self.metric)

            if not index:
                continue

            if len(index) > 1:
                subset = valmetrics[valmetrics.index.isin(index)]
                index = _bestidx(subset, method, self.metric)

            indexes.append(index[0])

        df = self.metrics[self.metrics.index.isin(indexes)]
        return df

    def _getpredictions(self, id, split="test", iteration=False):
        for pr in self.prs:
            if pr.id == id:
                df = pr.df[pr.df.split == split]
                if iteration:
                    data = {
                        "yhat": df.yhat, "ytrue": df.ytrue, "iteration": df.iteration
                    }
                else:
                    data = {"yhat": df.yhat, "ytrue": df.ytrue}
                df = pd.DataFrame(data=data, index=df.index)
                df.index = pd.to_datetime(df.index)
                return df

    def evaluate(self, overview=True, detailed=True):
        output = f"{self.output}/output.pdf"
        with matplotlib.backends.backend_pdf.PdfPages(output) as pdf:
            if overview:
                self.barofbests(pdf)
                self.lineofbests(pdf)
                self.errorovertime(pdf, "val")
                self.errorovertime(pdf, "test")
                self.lineofbest(pdf)

            if detailed:
                pass
                # table()
                # scatter4subplot()

            # Temporary
            # self.weekly(pdf)
            self.monthly(pdf)

    def errorovertime(self, pdf, split):
        fig, ax = plt.subplots(1, 1)
        colors = sns.color_palette()
        labels = []

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            labels.append(filtered.index[0])
            dfpred = self._getpredictions(
                filtered.index[0], split=split, iteration=True
            )
            ls = [linestyles[i + 1][1][1], linestyles[0][1][1]]
            palette = [colors[i + 1], colors[0]]

            # Compute RMSE for each backtest iteration
            rmse = []
            iterations = dfpred.iteration.unique()
            for iteration in iterations:
                chunk = dfpred[dfpred.iteration == iteration]
                rmse.append(((chunk.ytrue - chunk.yhat) ** 2).mean() ** .5)

            # y = abs(dfpred.yhat - dfpred.ytrue)
            sns.lineplot(data=rmse, dashes=ls, palette=palette, ax=ax)

            # markers = ["v", "o"]
            # for i, line in enumerate(ax.get_lines()):
            #     line.set_marker(markers[i])

        plt.title(f"RMSE per backtest iteration")
        plt.legend(labels=labels)
        plot.wrapup(pdf)

    def barofbests(self, pdf):
        df = self._boem()
        index = df.index.unique().to_list()

        # axes = df.plot.bar(
        #     rot=0, grid=True, subplots=True, layout=(2, 2), title=title, hue="split"
        # )

        metrics = df.columns.to_list()
        metrics.pop()

        df = df.reset_index()
        df = df.rename(columns={"index": "model"})

        fig, axes = plt.subplots(2, 2)
        for metric, ax in zip(metrics, axes.flatten()):
            subset = df[["model", metric, "split"]]
            sns.barplot(x="model", y=metric, hue="split", data=subset, ax=ax)

        for i, ax in enumerate(axes.flatten()):
            if i < 2:
                ax.set_xticks([])

            # ax.grid(axis="y", zorder=0)
            ax.set(xlabel=None)

            for container in ax.containers:
                ax.bar_label(container)

        title = f"Model with best {self.metric.upper()} of each method"
        fig.suptitle(title)

        plot.wrapup(pdf)

    def lineofbests(self, pdf):
        fig, ax = plt.subplots(1, 1)
        colors = sns.color_palette()
        labels = []

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            labels.append(filtered.index[0])
            dfpred = self._getpredictions(filtered.index[0])
            ls = [linestyles[i + 1][1][1], linestyles[0][1][1]]
            palette = [colors[i + 1], colors[0]]

            sns.lineplot(data=dfpred, dashes=ls, palette=palette, ax=ax)

            # markers = ["v", "o"]
            # for i, line in enumerate(ax.get_lines()):
            #     line.set_marker(markers[i])

        plt.legend(labels=labels)
        plot.wrapup(pdf)

    def lineofbest(self, pdf):
        for method in self.methods:
            # Select the model with the best performance based on some metric
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
