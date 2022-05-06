import os

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import src.plot as plot


# TODO:
#  - Create a single lineofbests with all best models
#  - Add tables (research on google: how to plot table in python)
#  - Get methods from config?
#  - List comprehension in getpredictions
#  + Save historical data with ProcessorResults
#  + Add historical data to plots (?)

# TODO: Evaluation report
#  - Performance of an established metric on a validation dataset
#  - Plots such as precision-recall curves
#  - Operational statistics such as inference speed
#  - Examples where the model was most confidently incorrect
#  - Save all of the hyper-parameters used to train the model


class Sink:
    def __init__(self, prs, path="output/sink"):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        self.df = None
        self.prs = prs
        self.output = path

        self._prs2df()

    def _prs2df(self):
        # Processor results to dataframe
        idx, r2, mae, mse, rmse = [], [], [], [], []
        for pr in self.prs:
            _idx = pr.id
            _r2, _mae, _mse, _rmse = self.getmetrics(pr.ytrue, pr.yhat)

            idx.append(_idx)
            r2.append(_r2)
            mae.append(_mae)
            mse.append(_mse)
            rmse.append(_rmse)

        # Create dataframe
        data = {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}
        self.df = pd.DataFrame(data=data, index=idx)

        return self.df

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
        lr = self.df.filter(like="LR", axis="index")
        lr = lr[lr.r2 == lr.r2.max()]

        knn = self.df.filter(like="KNN", axis="index")
        knn = knn[knn.r2 == knn.r2.max()]

        svr = self.df.filter(like="SVR", axis="index")
        svr = svr[svr.r2 == svr.r2.max()]

        mlp = self.df.filter(like="MLP", axis="index")
        mlp = mlp[mlp.r2 == mlp.r2.max()]

        df = pd.concat([lr, knn, svr, mlp])

        title = "Model with max R2 of each method"
        df.plot.bar(rot=0, grid=True, subplots=True, layout=(2, 2), title=title)

        plot.wrapup(pdf)

    def lineofbests(self, pdf):
        tmp = []

        methods = ["LR", "KNN", "SVR", "MLP"]
        for method in methods:
            # Select the model with best perfomance based on some metric
            filtered = self.df.filter(like=method, axis="index")
            filtered = filtered[filtered.r2 == filtered.r2.max()]

            if filtered.empty:
                continue

            dfpred = self.getpredictions(filtered.index[0])
            tmp.append(dfpred)

            ls = ["--", "-"]
            ax = dfpred.plot(rot=0, grid=True, style=ls)

            # markers = ["v", "o"]
            # for i, line in enumerate(ax.get_lines()):
            #     line.set_marker(markers[i])

            plt.title(f"Best {method} method: {filtered.index[0]}")

            # ==========================================================================
            # FIXME: worst and best are not of type datetime when calling dfpred.index[]
            # abserror = abs(dfpred.ytrue - dfpred.yhat)
            # idx = np.argsort(abserror)
            #
            # n = 10
            # best = idx[:n].to_list()
            # worst = np.flip(idx)[:n].to_list()
            #
            # plt.plot(
            #     dfpred.index[worst], dfpred.yhat[worst], linestyle="None", marker="o", color="tab:red"
            # )
            # plt.plot(
            #     dfpred.index[best], dfpred.yhat[best], linestyle="None", marker="o", color="tab:green"
            # )
            # ==========================================================================

            plot.wrapup(pdf)

    def getbestmodel(self):
        methods = ["LR", "KNN", "SVR", "MLP"]
        for method in methods:
            # Select the model with best perfomance based on some metric
            filtered = self.df.filter(like=method, axis="index")
            filtered = filtered[filtered.r2 == filtered.r2.max()]

            if filtered.empty:
                continue

            yield method, self.getpredictions(filtered.index[0])

    def weekly(self, pdf):
        for method, df in self.getbestmodel():
            mondays = np.where(df.index.weekday == 0)[0]
            mondays = np.random.choice(mondays, 4)
            fig, axes = plt.subplots(2, 2)

            for ax, startrow in zip(axes.flatten(), mondays):
                endrow = startrow + 7
                chunk = df.iloc[startrow:endrow, :]

                ls = ["--", "-"]
                title = f"{chunk.index[0].date()} to {chunk.index[-1].date()}"
                chunk.index = chunk.index.day_name()
                chunk.plot(rot=0, grid=True, style=ls, ax=ax, title=title)

            markers = ["v", "o"]
            for i, ax in enumerate(axes.flatten()):
                if i < 2:
                    ax.set_xticklabels([])

                for j, line in enumerate(ax.get_lines()):
                    line.set_marker(markers[j])

            fig.suptitle(f"Random weekly predictions")
            plot.wrapup(pdf)

    def monthly(self, pdf):
        for method, df in self.getbestmodel():
            firstday = np.where(df.index.day == 1)[0]
            firstday = np.random.choice(firstday, 4)
            fig, axes = plt.subplots(2, 2)

            for ax, startrow in zip(axes.flatten(), firstday):
                endrow = startrow + 30
                chunk = df.iloc[startrow:endrow, :]

                ls = ["--", "-"]
                title = f"{chunk.index[0].strftime('%b')}"
                chunk.index = chunk.index.day_name()
                chunk.plot(rot=90, grid=True, style=ls, ax=ax, title=title)

                ax.set_xticks(np.arange(len(chunk.index)), chunk.index, rotation=90)

            markers = ["v", "o"]
            for i, ax in enumerate(axes.flatten()):
                # if i < 2:
                #     ax.set_xticklabels([])

                for j, line in enumerate(ax.get_lines()):
                    line.set_marker(markers[j])

            fig.suptitle(f"Random monthly predictions")
            plot.wrapup(pdf)

    def getpredictions(self, id):
        for pr in self.prs:
            if pr.id == id:
                data = {"yhat": pr.yhat, "ytrue": pr.ytrue}
                df = pd.DataFrame(data=data, index=pr.timestamp)
                df.index = pd.to_datetime(df.index)
                return df

    def getmetrics(self, ytrue, yhat):
        r2 = r2_score(ytrue, yhat)
        mae = mean_absolute_error(ytrue, yhat)
        mse = mean_squared_error(ytrue, yhat, squared=True)
        rmse = mean_squared_error(ytrue, yhat, squared=False)

        return r2, mae, mse, rmse
