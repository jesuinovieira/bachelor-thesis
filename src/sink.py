import datetime
import os

import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import ConnectionPatch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from tabulate import tabulate

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


def _getmetrics(ytrue, yhat):
    r2 = r2_score(ytrue, yhat)
    mae = mean_absolute_error(ytrue, yhat)
    mape = mean_absolute_percentage_error(ytrue, yhat)
    rmse = mean_squared_error(ytrue, yhat, squared=False)

    return r2, mae, mape, rmse


def _getdfmetrics(df, average):
    iterations = df.iteration.unique()
    scoring = dict(r2=[], mae=[], mape=[], rmse=[])

    # Compute error for each backtest iteration
    for iteration in iterations:
        chunk = df[df.iteration == iteration]
        r2, mae, mape, rmse = _getmetrics(chunk.ytrue, chunk.yhat)

        scoring["r2"].append(r2)
        scoring["mae"].append(mae)
        scoring["mape"].append(mape)
        scoring["rmse"].append(rmse)

    if average:
        r2 = np.mean(scoring["r2"])
        mae = np.mean(scoring["mae"])
        mape = np.mean(scoring["mape"])
        rmse = np.mean(scoring["rmse"])
        return r2, mae, mape, rmse
    else:
        return scoring


def _bestidx(df, method, metric):
    df = df.filter(like=method, axis="index")

    condition = df[metric].min() if metric != "r2" else df[metric].max()
    df = df[df[metric] == condition]

    return df.index.to_list()


class Sink:
    def __init__(self, prs, path):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        # FIXME: hardcoded. Last iteration has only one sample
        for pr in prs:
            pr.df.at["2018-12-31", "iteration"] = pr.df.iteration.unique()[-2]
            pr.df.at["2019-12-31", "iteration"] = pr.df.iteration.unique()[-2]

        self.singlepdf = False

        self.prs = prs
        self.output = path
        self.metric = "rmse"
        self.methods = ["LR", "KNN", "SVR", "MLP"]

        self.metrics = self._prs2df()

        # metrics-method.csv
        dst = os.path.join(self.output, "metrics-method.csv")
        self.metrics.to_csv(dst, index_label="model")

        # metrics-rmse.csv
        dst = os.path.join(self.output, "metrics-rmse.csv")
        tmp = self.metrics.sort_values(["split", "rmse"])
        tmp.to_csv(dst, index_label="model")

        # metrics-backtesting.csv
        models = self.metrics.index.to_series().str.split('-')
        models = models.to_list()
        models = [f"{item[1]}-{item[2]}" for item in models]
        tmp = self.metrics
        tmp["tmp"] = models
        tmp.sort_values(["split", "tmp"])
        tmp = tmp.drop("tmp", axis=1)
        dst = os.path.join(self.output, "metrics-backtesting.csv")
        tmp.to_csv(dst, index_label="model")

    def _prs2df(self):
        index = []
        data = dict(r2=[], mae=[], mape=[], rmse=[], split=[])
        for split in ["val", "test"]:
            for pr in self.prs:
                df = pr.df[pr.df.split == split]

                tmp = df[["ytrue", "yhat", "iteration"]]
                r2, mae, mape, rmse = _getdfmetrics(tmp, average=True)

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

    def _save(self, pdf, filename):
        if not pdf:
            with matplotlib.backends.backend_pdf.PdfPages(filename) as pdf:
                plot.wrapup(pdf)
        else:
            plot.wrapup(pdf)

    def _evaluate(self, pdf=None):
        self.realvspredicted(pdf)
        # self.predictions42months(pdf, 9, 30, 12, 31)
        # self.predictions42months__(pdf, 9, 30, 12, 31)
        # self.errorbymonth4bests2(pdf, "test")

        # self.metrics4bests(pdf)

        # self.metrics4each("val")
        # self.metrics4each("test")

        # self.predictions4bests(pdf)
        # self.predictions4months(pdf, 9, 30)
        # self.predictions4months(pdf, 12, 31)

        # self.monthly4bests(pdf, [9, 12])
        # self.errorbymonth4bests1(pdf, "test")

        # self.errorbyiteration4bests(pdf, "test")

        # Currently not used
        # ------------------------------------------------------------------------------
        #
        # self.variabilitybymonth(pdf)
        # self._predictions4months(pdf, 9, 12)
        # self._variabilitybymonth(pdf)
        #
        # self.corr4bests(pdf)
        # self.errorbymonth4bests(pdf, "val")
        # self.errorbyiteration4bests(pdf, "val")
        # self.monthly4each(pdf)
        # self.weekly4each(pdf)

    def evaluate(self):
        if self.singlepdf:
            output = f"{self.output}/output.pdf"
            with matplotlib.backends.backend_pdf.PdfPages(output) as pdf:
                self._evaluate(pdf)
        else:
            self._evaluate()

    # Dev
    # ==================================================================================

    def predictions42months(self, pdf, upperm, upperld, lowerm, lowerld):
        # Set figure size as a fraction of the column width
        hf = plot.textheigth / plot.textwidth * 0.9
        figsize = plot.get_figsize(plot.textwidth, wf=1.1, hf=hf)
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on defined metric
            filtered = self._getbest(method)
            if filtered.empty:
                continue

            # Get predictions
            dfpred = self._getpredictions(filtered.index[0])

            # Plot ytrue only first time
            if i == 0:
                for j, ax in enumerate(axes):
                    label = "ytrue" if j == 0 else None
                    color = plot.palette[i]
                    # FIXME: line style
                    ls = linestyles[0][1][1]
                    sns.lineplot(
                        x=dfpred.index, y=dfpred.ytrue, label=label,
                        color=color, ax=ax, lw=1.5 if j != 1 else 1
                    )
                    # print(f"[ax{j}] coloridx[i]={i}, label={label}")

            # Plot yhat for the current model
            for j, ax in enumerate(axes):
                label = filtered.index[0] if j == 0 else None
                color = plot.palette[i + 1]
                # FIXME: line style
                ls = linestyles[i + 1][1][1]
                sns.lineplot(
                    x=dfpred.index, y=dfpred.yhat, label=label,
                    color=color, ax=ax, lw=1.5 if j != 1 else 1
                )
                # print(f"[ax{j}] coloridx={i + 1}, label={label}")

            # FIXME: improve margin of zoomed plots
            margins = 0.05

            # Upper plot zoom
            xmin = datetime.date(2019, upperm, 1)
            xmax = datetime.date(2019, upperm, upperld)
            axes[0].set_xlim([xmin, xmax])

            ymin = dfpred.ytrue[xmin:xmax].min()
            ymin = ymin - ymin * margins
            ymax = dfpred.ytrue[xmin:xmax].max()
            ymax = ymax + ymax * margins
            axes[0].set_ylim([ymin, ymax])

            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # Lower plot zoom
            xmin = datetime.date(2019, lowerm, 1)
            xmax = datetime.date(2019, lowerm, lowerld)
            axes[2].set_xlim([xmin, xmax])

            ymin = dfpred.ytrue[xmin:xmax].min()
            ymin = ymin - ymin * margins
            ymax = dfpred.ytrue[xmin:xmax].max()
            ymax = ymax + ymax * margins
            axes[2].set_ylim([ymin, ymax])

            axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # FIXME: can't handle x-axis as timestamp? Use integer x-axis and later
            #  change to timestamp
            # conn1 = ConnectionPatch(
            #     xyA=(dfpred.index[10], dfpred.ytrue[10]),
            #     xyB=(dfpred.index[10], dfpred.ytrue[10]),
            #     coordsA=axes[1].transData,
            #     coordsB=axes[0].transData,
            #     axesA=axes[1],
            #     axesB=axes[0],
            #     arrowstyle="-"
            # )
            #
            # axes[1].add_artist(conn1)

        for ax in axes:
            # ax.set_ylabel(None)
            ax.set_ylabel("Water demand (m\u00b3)")
            ax.set_xlabel("Timestamp")

            ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        handles, labels = axes[0].get_legend_handles_labels()
        # order = [1, 0, 2, 3, 4]
        # handles, labels = [handles[k] for k in order], [labels[k] for k in order]
        labels[0] = "Observed"

        # Remove long identifier
        labels = [s.split("1")[0] for s in labels]

        axes[0].legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.175), ncol=5, prop={"size": 8}
        )

        filename = f"{self.output}/predictions42months.pdf"
        self._save(pdf, filename)

    def predictions42months__(self, pdf, upperm, upperld, lowerm, lowerld):
        # Set figure size as a fraction of the column width
        hf = 0.5
        figsize = plot.get_figsize(plot.textwidth, wf=1.5, hf=hf)
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on defined metric
            filtered = self._getbest(method)
            if filtered.empty:
                continue

            # Get predictions
            dfpred = self._getpredictions(filtered.index[0])

            # Plot ytrue only first time
            if i == 0:
                for j, ax in enumerate(axes):
                    label = "ytrue" if j == 0 else None
                    color = plot.palette[i]
                    # FIXME: line style
                    ls = linestyles[0][1][1]
                    sns.lineplot(
                        x=dfpred.index, y=dfpred.ytrue, label=label,
                        color=color, ax=ax, lw=1.75 if j != 1 else 1
                    )
                    # print(f"[ax{j}] coloridx[i]={i}, label={label}")

            # Plot yhat for the current model
            for j, ax in enumerate(axes):
                label = filtered.index[0] if j == 0 else None
                color = plot.palette[i + 1]
                # FIXME: line style
                ls = linestyles[i + 1][1][1]
                sns.lineplot(
                    x=dfpred.index, y=dfpred.yhat, label=label,
                    color=color, ax=ax, lw=1.15 if j != 1 else 1,
                    # linestyle=ls
                )
                # print(f"[ax{j}] coloridx={i + 1}, label={label}")

            # FIXME: improve margin of zoomed plots
            margins = 0.05

            # Upper plot zoom
            # xmin = datetime.date(2019, upperm, 1)
            # xmax = datetime.date(2019, upperm, upperld)
            # axes[0].set_xlim([xmin, xmax])
            #
            # ymin = dfpred.ytrue[xmin:xmax].min()
            # ymin = ymin - ymin * margins
            # ymax = dfpred.ytrue[xmin:xmax].max()
            # ymax = ymax + ymax * margins
            # axes[0].set_ylim([ymin, ymax])
            #
            # axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # Lower plot zoom
            xmin = datetime.date(2019, lowerm, 1)
            xmax = datetime.date(2019, lowerm, lowerld)
            axes[0].set_xlim([xmin, xmax])

            ymin = dfpred.ytrue[xmin:xmax].min()
            ymin = ymin - ymin * margins
            ymax = dfpred.ytrue[xmin:xmax].max()
            ymax = ymax + ymax * margins
            axes[0].set_ylim([ymin, ymax])

            axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # FIXME: can't handle x-axis as timestamp? Use integer x-axis and later
            #  change to timestamp
            # conn1 = ConnectionPatch(
            #     xyA=(dfpred.index[10], dfpred.ytrue[10]),
            #     xyB=(dfpred.index[10], dfpred.ytrue[10]),
            #     coordsA=axes[1].transData,
            #     coordsB=axes[0].transData,
            #     axesA=axes[1],
            #     axesB=axes[0],
            #     arrowstyle="-"
            # )
            #
            # axes[1].add_artist(conn1)

        for ax in axes:
            # ax.set_ylabel(None)
            ax.set_ylabel("Water demand (m\u00b3)")
            ax.set_xlabel("Timestamp")

            ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        handles, labels = axes[0].get_legend_handles_labels()
        # order = [1, 0, 2, 3, 4]
        # handles, labels = [handles[k] for k in order], [labels[k] for k in order]
        labels[0] = "Observed"

        # Remove long identifier
        labels = [s.split("1")[0] for s in labels]

        axes[0].legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.115), ncol=5, prop={"size": 8}
        )

        filename = f"{self.output}/predictions42months__.pdf"
        self._save(pdf, filename)

    def metrics4each(self, split):
        print(f"\nSet: {split}\n")

        data = []
        for i, pr in enumerate(self.prs):
            dfpred = self._getpredictions(pr.id, split=split, iteration=True)
            scoring = _getdfmetrics(dfpred, average=False)

            _data = [
                pr.id,

                round(np.mean(scoring["rmse"]), 2),
                round(np.std(scoring["rmse"]), 3),

                round(np.mean(scoring["mae"]), 2),
                round(np.std(scoring["mae"]), 3),

                round(np.mean(scoring["mape"]) * 100, 2),
                round(np.std(scoring["mape"]) * 100, 3),

                round(np.mean(scoring["r2"]), 2),
                round(np.std(scoring["r2"]), 3),
            ]

            data.append(_data)

        myorder = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
        data = [data[i] for i in myorder]
        sigma = "\u03C3"

        headers = ["Model", "RMSE", sigma, "MAE", sigma, "MAPE", sigma, "R2", sigma]
        print(tabulate(data, headers=headers))

    def realvspredicted(self, pdf):
        # hf = plot.textheigth / plot.textwidth * 0.25
        figsize = plot.get_figsize(plot.textwidth, wf=1.0, hf=1.0)
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for i, method in enumerate(self.methods):
            ax = axes.flatten()[i]

            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)
            if filtered.empty:
                continue

            dfpred = self._getpredictions(filtered.index[0])
            model = f"{filtered.index[0].split('1')[0]}"

            # color = plot.palette[i + 1]
            sns.regplot(
                x="ytrue", y="yhat", data=dfpred, ax=ax,
                scatter=True, label=model, scatter_kws={"alpha": 0.5}
                # color=color,
            )

            corr = dfpred.corr()
            corr = round(corr.iloc[1, 0], 2)
            r2 = round(r2_score(dfpred.ytrue, dfpred.yhat), 2)
            r = "$\it{r}$"
            ax.set_title(f"{model}: {r}={corr}, $R$\u00b2={r2}")
            ax.set_title(f"{model}: $R$\u00b2={r2}")

        for i, ax in enumerate(axes.flatten()):
            ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            ax.xaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

            if i == 0 or i == 2:
                ax.set(ylabel="Predicted")
            else:
                ax.set(ylabel=None)
                # ax.set_yticklabels([])

            if i < 2:
                ax.set_xticklabels([])
                ax.set(xlabel=None)
            else:
                ax.set(xlabel="Observed")

        filename = f"{self.output}/real-vs-predicted.pdf"
        self._save(pdf, filename)

    def realvspredicted_bkp(self, pdf):
        # hf = plot.textheigth / plot.textwidth * 0.4
        figsize = plot.get_figsize(plot.textwidth, wf=1.0, hf=1.0)
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        for i, method in enumerate(self.methods):
            ax = axes.flatten()[i]

            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)
            if filtered.empty:
                continue

            dfpred = self._getpredictions(filtered.index[0])
            model = f"{filtered.index[0].split('1')[0]}"

            # color = plot.palette[i + 1]
            sns.regplot(
                x="ytrue", y="yhat", data=dfpred, ax=ax,
                scatter=True, label=model
                # color=color,
            )

            corr = dfpred.corr()
            corr = round(corr.iloc[1, 0], 2)
            r2 = round(r2_score(dfpred.ytrue, dfpred.yhat), 2)
            r = "$\it{r}$"
            ax.set_title(f"{model}: {r}={corr}, $R$\u00b2={r2}")
            ax.set_title(f"{model}: $R$\u00b2={r2}")

        for i, ax in enumerate(axes.flatten()):
            ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            ax.xaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))

            if i == 0 or i == 2:
                ax.set(ylabel="Predicted")
            else:
                ax.set(ylabel=None)
                # ax.set_yticklabels([])

            if i < 2:
                ax.set_xticklabels([])
                ax.set(xlabel=None)
            else:
                ax.set(xlabel="Observed")

        filename = f"{self.output}/real-vs-predicted.pdf"
        self._save(pdf, filename)

    # Currently used
    # ==================================================================================

    def metrics4bests(self, pdf):
        df = self._boem()

        metrics = df.columns.to_list()
        metrics.pop()

        df = df.reset_index()
        df = df.rename(columns={"index": "model"})

        DPI = 200
        figsize = (1920 / DPI, 1080 / DPI)
        # figsize = (6.299212813062128, 3.893127620841233)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for metric, ax in zip(metrics, axes.flatten()):
            subset = df[["model", metric, "split"]]
            sns.barplot(x="model", y=metric, hue="split", data=subset, ax=ax)

        for i, ax in enumerate(axes.flatten()):
            # TODO: fix legend to dont overlap plot
            ax.get_legend().remove()

            if i < 2:
                ax.set_xticks([])

            ax.set(xlabel=None)
            for container in ax.containers:
                ax.bar_label(container)

        title = f"Model with best {self.metric.upper()} of each method"
        fig.suptitle(title)

        filename = f"{self.output}/4bests-metrics.pdf"
        self._save(pdf, filename)

    def predictions4bests(self, pdf):
        fig, ax = plt.subplots(1, 1)
        colors = sns.color_palette()

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            dfpred = self._getpredictions(filtered.index[0])
            ls = [linestyles[i + 1][1][1], linestyles[0][1][1]]
            palette = [colors[i + 1], colors[0]]

            dfpred = dfpred.rename(columns={"yhat": filtered.index[0]})
            sns.lineplot(data=dfpred, dashes=ls, palette=palette, ax=ax)

            # markers = ["v", "o"]
            # for i, line in enumerate(ax.get_lines()):
            #     line.set_marker(markers[i])

        # Remove duplicated labels (ytrue)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        filename = f"{self.output}/4bests-predictions.pdf"
        self._save(pdf, filename)

    def predictions4months(self, pdf, month, lastday):
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

        import matplotlib as mpl
        mpl.rcParams["lines.linewidth"] = 1
        DPI = 200

        fig = plt.figure()
        # fig = plt.figure(figsize=(1920 / DPI, 1080 / DPI))
        # fig = plt.figure(figsize=(6, 5))
        # fig = plt.figure(figsize=(6.299212813062128, 3.893127620841233))

        plt.subplots_adjust(bottom=0., left=0., top=1., right=1.)

        # add_subplot(row, column, cell)
        sub1 = fig.add_subplot(2, 1, 1)
        sub2 = fig.add_subplot(2, 1, 2)
        # sub3 = fig.add_subplot(3, 1, 3)

        colors = sns.color_palette()

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            dfpred = self._getpredictions(filtered.index[0])
            ls = [linestyles[i + 1][1][1], linestyles[0][1][1]]
            palette = [colors[i + 1], colors[0]]

            if i == 0:
                sns.lineplot(
                    x=dfpred.index, y=dfpred.ytrue, label="ytrue",
                    dashes=ls, palette=colors[0], ax=sub1
                )
                sns.lineplot(
                    x=dfpred.index, y=dfpred.ytrue, label="ytrue",
                    dashes=ls, palette=colors[0], ax=sub2
                )

            sns.lineplot(
                x=dfpred.index, y=dfpred.yhat, label=filtered.index[0],
                dashes=ls, color=colors[i + 1], ax=sub2
            )

            # Zoom plots

            sns.lineplot(
                x=dfpred.index, y=dfpred.yhat, label=filtered.index[0],
                dashes=ls, color=colors[i + 1], ax=sub1
            )
            sub1.set_xlim(
                [datetime.date(2019, month, 1), datetime.date(2019, month, lastday)]
            )
            sub1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

            # sns.lineplot(
            #     x=dfpred.index, y=dfpred.yhat, dashes=ls,
            #     palette=palette, ax=sub3
            # )
            # sub3.set_xlim([datetime.date(2019, 12, 1), datetime.date(2019, 12, 31)])
            # sub3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        # box = sub1.get_position()
        # sub1.set_position([
        #     box.x0, box.y0 + box.height * 0.1,
        #     box.width, box.height * 0.9
        # ])

        # for j, line in enumerate(sub1.lines):
        #     ls_ = linestyles[j + 1][1][1]
        #     line.set_linestyle(ls_)

        sub1.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.05), fancybox=True,
            shadow=False, ncol=5
        )
        # plt.show()

        filename = f"{self.output}/predictions4months_.pdf"
        self._save(pdf, filename)

    def monthly4bests(self, pdf, months):
        colors = sns.color_palette()

        for month in months:
            fig, ax = plt.subplots()

            for i, method in enumerate(self.methods):
                filtered = self._getbest(method)
                if filtered.empty:
                    continue

                predictions = self._getpredictions(filtered.index[0])
                chunk = predictions[predictions.index.month.isin([month])]

                ls = [linestyles[i + 1][1][1], linestyles[0][1][1]]
                palette = [colors[i + 1], colors[0]]

                chunk = chunk.rename(columns={"yhat": filtered.index[0]})
                sns.lineplot(data=chunk, dashes=ls, palette=palette, ax=ax)
                plt.xticks(rotation=45)

                # chunk.index = chunk.index.day_name().str.slice(0, 3)
                # ax.set_xticks(np.arange(len(chunk.index)), chunk.index, rotation=90)

            # Remove duplicated labels (ytrue)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            filename = f"{self.output}/4bests-predictions-monthly.pdf"
            self._save(pdf, filename)

    def errorbymonth4bests1(self, pdf, split):
        fig, axes = plt.subplots(2, 2)

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            dfpred = self._getpredictions(
                filtered.index[0], split=split, iteration=True
            )

            dfpred["month"] = dfpred.index.strftime("%b")
            dfpred["error"] = abs(dfpred.yhat - dfpred.ytrue)

            tmp = dfpred.groupby("month")["error"].mean()
            tmp = tmp.sort_values()

            ax = axes.flatten()[i]
            color = sns.color_palette()[0]
            meanprops = {
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "#4c4c4c",
                "markersize": "3"
            }

            sns.boxplot(
                x="month", y="error", data=dfpred, ax=ax, color=color, showmeans=True,
                meanprops=meanprops
            )
            ax.set_title(
                f"worst: {tmp.index[-1]}={round(tmp[-1], 2)}, "
                f"{tmp.index[-2]}={round(tmp[-2], 2)} [{method}]\n"
                f"best: {tmp.index[0]}={round(tmp[0], 2)}, "
                f"{tmp.index[1]}={round(tmp[1], 2)} [{method}]"
            )

        for i, ax in enumerate(axes.flatten()):
            if i < 2:
                ax.set_xticklabels([])
            ax.set(xlabel=None)

        # fig.suptitle(f"Error by month [{split}]")

        filename = f"{self.output}/4bests-error-by-month.pdf"
        self._save(pdf, filename)

    def errorbymonth4bests2(self, pdf, split):
        # Set figure size as a fraction of the column width
        figsize = plot.get_figsize(plot.textwidth, wf=1.0)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # DPI = 200
        # fig, ax = plt.subplots(figsize=(1920 / DPI, 1080 / DPI))
        df = None

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            dfpred = self._getpredictions(
                filtered.index[0], split=split, iteration=True
            )

            dfpred["month"] = dfpred.index.strftime("%b")
            dfpred["error"] = abs(dfpred.yhat - dfpred.ytrue)
            dfpred["method"] = filtered.index[0]

            dfpred = dfpred.drop(["yhat", "ytrue", "iteration"], axis=1)

            if df is None:
                df = dfpred
            else:
                df = pd.concat([df, dfpred])

        palette = plot.palette[1:5]
        meanprops = {
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#4c4c4c",
            "markersize": "3"
        }

        sns.boxplot(
            x="month", y="error", data=df, hue="method", ax=ax, palette=palette,
            showmeans=True, meanprops=meanprops
        )

        ax.set_ylabel("Absolute error")
        ax.set_xlabel("Month")

        ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        handles, labels = ax.get_legend_handles_labels()
        labels = [s.split("1")[0] for s in labels]
        ax.legend(handles=handles, labels=labels, title=None)

        filename = f"{self.output}/4bests-error-by-month_.pdf"
        self._save(pdf, filename)

    def errorbyiteration4bests(self, pdf, split):
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

            # Compute error for each backtest iteration
            data = dict(r2=[], mae=[], mape=[], rmse=[])
            iterations = dfpred.iteration.unique()
            for iteration in iterations:
                chunk = dfpred[dfpred.iteration == iteration]

                _rmse = mean_squared_error(chunk.ytrue, chunk.yhat, squared=False)
                _mae = mean_absolute_error(chunk.ytrue, chunk.yhat)

                r2, mae, mape, rmse = _getmetrics(chunk.ytrue, chunk.yhat)

                # Last chunk has only one..
                if iteration != iterations[-1]:
                    data["r2"].append(r2)

                data["mae"].append(mae)
                data["mape"].append(mape)
                data["rmse"].append(rmse)

            # TODO: fix line style
            ls = [linestyles[i + 1][1][1]]
            palette = colors[i + 1]
            sns.lineplot(data=data["mae"], dashes=ls, color=palette, ax=ax)

        plt.title(f"MAE per backtest iteration [{split}]")
        plt.legend(labels=labels)

        filename = f"{self.output}/4bests-error-by-iteration.pdf"
        self._save(pdf, filename)

    # Currently not used
    # ==================================================================================

    def variabilitybymonth(self, pdf):
        df = self.prs[0].df
        for month in range(12):
            month += 1
            fig, ax = plt.subplots()
            for year in [2016, 2017, 2018, 2019]:
                tmp = df[df.index.year == year]
                tmp = tmp[tmp.index.month == month]

                tmp.index = tmp.index.strftime("%m-%d")
                sns.lineplot(
                    x=tmp.index, y=tmp.water_produced,  # dashes=ls,
                    palette=sns.color_palette(), ax=ax
                )
                ax.set(title=f"{month}")
                plt.xticks(rotation=90)
                ax.set_ylim([0, 1])

            filename = f"{self.output}/variability-by-month-{month}.pdf"
            self._save(pdf, filename)

    def _variabilitybymonth(self, pdf):
        df = self.prs[0].df
        for month in range(12):
            month += 1
            fig, axes = plt.subplots(2, 1)
            for year in [2016, 2017, 2018, 2019]:
                tmp = df[df.index.year == year]
                tmp = tmp[tmp.index.month == month]

                tmp.index = tmp.index.strftime("%m-%d")
                sns.lineplot(
                    x=tmp.index, y=tmp.water_produced,  # dashes=ls,
                    palette=sns.color_palette(), ax=axes[0]
                )
                axes[0].set(title=f"{month}")
                plt.xticks(rotation=90)
                axes[0].set_ylim([0, 1])

                if year != 2019:
                    continue

                sns.lineplot(
                    x=tmp.index, y=tmp.relative_humidity_mean,  # dashes=ls,
                    palette=sns.color_palette(), ax=axes[1], label="humidity"
                )
                sns.lineplot(
                    x=tmp.index, y=tmp.radiation_mean,  # dashes=ls,
                    palette=sns.color_palette(), ax=axes[1], label="radiation"
                )
                sns.lineplot(
                    x=tmp.index, y=tmp.precipitation_mean,  # dashes=ls,
                    palette=sns.color_palette(), ax=axes[1], label="precipitation"
                )
                sns.lineplot(
                    x=tmp.index, y=tmp.temperature_mean,  # dashes=ls,
                    palette=sns.color_palette(), ax=axes[1], label="temperature"
                )

            filename = f"{self.output}/variability-by-month-{month}_.pdf"
            self._save(pdf, filename)

    def _predictions4months(self, pdf, m1, m2):
        # https://medium.com/the-stem/3-minute-guide-to-use-subplots-connection-patch-in-matplotlib-fe50ac0fbeb8
        # https://regenerativetoday.com/some-tricks-to-make-matplotlib-visualization-even-better/

        fig = plt.figure(figsize=(6, 5))
        plt.subplots_adjust(bottom=0., left=0, top=1., right=1)

        # two rows, two columns, fist cell
        sub1 = fig.add_subplot(2, 2, 1)
        # two rows, two columns, second cell
        sub2 = fig.add_subplot(2, 2, 2)
        # two rows, two colums, combined third and fourth cell
        sub3 = fig.add_subplot(2, 2, (3, 4))

        colors = sns.color_palette()

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            if filtered.empty:
                continue

            dfpred = self._getpredictions(filtered.index[0])
            ls = [linestyles[i + 1][1][1], linestyles[0][1][1]]
            palette = [colors[i + 1], colors[0]]

            dfpred = dfpred.rename(columns={"yhat": filtered.index[0]})
            sns.lineplot(data=dfpred, dashes=ls, palette=palette, ax=sub3)

            # Zoom plots

            sns.lineplot(data=dfpred, dashes=ls, palette=palette, ax=sub1)
            sub1.set_xlim([datetime.date(2019, 9, 1), datetime.date(2019, 9, 30)])
            sub1.get_legend().remove()
            sub1.set_xticks([])

            sns.lineplot(data=dfpred, dashes=ls, palette=palette, ax=sub2)
            sub2.set_xlim([datetime.date(2019, 12, 1), datetime.date(2019, 12, 31)])
            sub2.get_legend().remove()
            sub2.set_xticks([])

            # Connection patch

            # Create left side of Connection patch for first axes
            # con1 = ConnectionPatch(xyA=(datetime.date(2019, 9, 1), 25000), coordsA=sub1.transData,
            #                        xyB=(datetime.date(2019, 9, 1), 25000), coordsB=sub3.transData,
            #                        color='green')  # (1, 0.755) cordinate point lhs
            # # Add left side to the figure
            # fig.add_artist(con1)

            # # Create right side of Connection patch for first axes
            # con2 = ConnectionPatch(xyA=(2, 1.1), coordsA=sub1.transData,
            #                        xyB=(2, 1.1), coordsB=sub3.transData, color='green')
            # # Add right side to the figure
            # fig.add_artist(con2)

            # # Create left side of Connection patch for second axes
            # con3 = ConnectionPatch(xyA=(5, -0.79), coordsA=sub2.transData,
            #                        xyB=(5, -0.79), coordsB=sub3.transData, color='orange')
            # # Add left side to the figure
            # fig.add_artist(con3)
            #
            # # Create right side of Connection patch for second axes
            # con4 = ConnectionPatch(xyA=(6, -0.85), coordsA=sub2.transData,
            #                        xyB=(6, -0.85), coordsB=sub3.transData, color='orange')
            # # Add right side to the figure
            # fig.add_artist(con4)

        # Remove duplicated labels (ytrue)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        filename = f"{self.output}/predictions4months.pdf"
        self._save(pdf, filename)

    def corr4bests(self, pdf):
        DPI = 200
        rcParams["figure.figsize"] = (1920 / DPI, 1080 / DPI)
        rcParams["font.size"] = "6"

        for i, method in enumerate(self.methods):
            # Select the model with the best performance based on some metric
            filtered = self._getbest(method)

            df = None
            for pr in self.prs:
                if pr.id == filtered.index[0]:
                    df = pr.df

            df = df[df.index.year == 2019]
            df["error"] = abs(df.ytrue - df.yhat)
            df = df.drop(columns=["year", "ytrue", "yhat"])

            corr = df.corr().sort_values("error", ascending=True)
            corr = corr.reindex(corr.index, axis=1)

            plot.corrmatrix(corr, f"{filtered.index[0]}", pdf)

            filename = f"{self.output}/4bests-correlation-{i}.pdf"
            self._save(pdf, filename)

    def errorbyseason4bests(self, pdf, split):
        # TODO
        pass

    def prediction4each(self, pdf):
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

            filename = f"{self.output}/{method.lower()}-prediction.pdf"
            self._save(pdf, filename)

    def monthly4each(self, pdf):
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

            filename = f"{self.output}/{method.lower()}-monthly.pdf"
            self._save(pdf, filename)

    def weekly4each(self, pdf):
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

            filename = f"{self.output}/{method.lower()}-weekly.pdf"
            self._save(pdf, filename)
