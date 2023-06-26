import dateutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import src.plot
from src import plot


def read(src):
    df = pd.read_csv(src)
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")

    return df


def outliers(series, k=1.5):
    # Find outlier using first and third quartiles and interquartile range.
    q1, q3 = series.quantile(.25), series.quantile(.75)
    iqr = q3 - q1
    lb, ub = q1 - k * iqr, q3 + k * iqr

    idxs = (series < lb) | (series > ub)

    return idxs


def everything(df, city, pdf, show):
    # Plot all years
    fig, axs = plot.setup(nrows=1)

    # hf = ?
    figsize = plot.get_figsize(plot.textwidth, wf=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # df.plot(title=title, rot=0, grid=True, use_index=True, ax=axs)
    y = df[f"produced-{city}"]
    sns.lineplot(x=df.index, y=y, label="todo", ax=ax)
    y = df[f"consumed-{city}"]
    sns.lineplot(x=df.index, y=y, label="todo", ax=ax)

    # corr = df.corr().iloc[0, 1]
    # title = f"{city}\ncorr={round(corr, 2)}"
    axs.set_xlabel(None)

    plot.save(pdf, "output/eda/todo.pdf")


def yearly(df, city, pdf, show):
    years = [2016, 2017, 2018, 2019]

    # wf = 0.5 * 2
    # hf = (5. ** 0.5 - 1.0) / 2.0 * 2
    figsize = plot.get_figsize(plot.textwidth, wf=1.0)
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    i = 0
    for ax, year in zip(axes.flatten(), years):
        chunk = df.loc[f"{year}"]
        corr = chunk.corr().iloc[0, 1]

        length = len(chunk[f"consumed-{city}"]) - chunk[f"consumed-{city}"].isna().sum()

        r = "$\it{r}$"
        r2 = "$R^2$"
        title = f"{r}={round(corr, 2)}"

        figsize = plot.get_figsize(plot.textwidth, 1.0)
        chunk.plot(
            rot=90, grid=True, ax=ax, use_index=True, title=title, lw=1.5,
            figsize=figsize, fontsize=plot.SIZE
        )
        ax.legend(["Produced", "Consumed"])

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        df1 = chunk.drop(columns=["produced-gtba"])
        df2 = chunk.drop(columns=["consumed-gtba"])

        # sns.lineplot(data=chunk, ax=ax)
        # sns.lineplot(data=df2, ax=ax)
        # sns.lineplot(data=df1, ax=ax)

        # if i == 0:
        #     sns.lineplot(x=chunk.index, y="produced-gtba", data=chunk, ax=ax)
        #     sns.lineplot(x=chunk.index, y="consumed-gtba", data=chunk, ax=ax)
        #     # ax.legend(["Water produced", "Water consumed from the reservoirs"])
        # else:
        #     sns.lineplot(x=chunk.index, y="produced-gtba", data=chunk, ax=ax)
        #     sns.lineplot(x=chunk.index, y="consumed-gtba", data=chunk, ax=ax)

        i += 1

        ax.set_ylabel("Water demand (m\u00b3)")
        ax.set_xlabel("Timestamp")
        ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    fig.suptitle(None)
    for i, ax in enumerate(axes.flatten()):
        ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        if i < 2:
            ax.set_xticklabels([])
            ax.set(xlabel=None)
        if i == 1 or i == 3:
            # ax.set_yticklabels([])
            ax.set(ylabel=None)
        if i > 0:
            ax.get_legend().remove()

    # fig.tight_layout()

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    labels = ["Water produced in the WTPs", "Water consumed from the reservoirs"]

    fig.legend(
        handles, labels,
        loc="center", bbox_to_anchor=(0.5, 0.96), ncol=2, prop={"size": 8}
    )

    for i, ax in enumerate(axes.flatten()):
        if i == 0:
            ax.get_legend().remove()

    src.plot.save(pdf, "produced-vs-consumed.pdf")


def monthly(df, city, pdf, show):
    years = [2016, 2017, 2018, 2019]
    for year in years:
        fig, axes = plt.subplots(2, 2)
        for month in range(0, 12):
            i = month % 4
            if i == 0 and month != 0:
                fig.suptitle(f"{city} {year}")
                plot.wrapup(pdf, show)
                fig, axes = plt.subplots(2, 2)

            ax = axes.flatten()[i]

            month = str(month + 1).rjust(2, "0")
            chunk = df.loc[f"{year}-{month}"]
            corr = chunk.corr().iloc[0, 1]

            title = f"corr={round(corr, 2)} ({chunk.index[0].strftime('%b')})"
            chunk.index = chunk.index.day_name().str.slice(0, 3)
            chunk.plot(rot=90, grid=True, ax=ax, use_index=True, title=title)

            ax.set_xticks(np.arange(len(chunk.index)), chunk.index, rotation=90)
            ax.set_xlabel(None)

        fig.suptitle(f"{city} {year}")
        plot.wrapup(pdf, show)


def E1P(df, city, pdf, show):
    if len(df.columns) == 1:
        return

    # everything(df, city, pdf, show)
    yearly(df, city, pdf, show)
    # monthly(df, city, pdf, show)


def E23P(df, title, pdf, show):
    # TODO: move this plot to plot?
    # style = ["--", "--", "-", "-", "-"]
    years = [2016, 2017, 2018, 2019]

    # TODO: df.plot(subplots=True)
    # Plot all years
    fig, axs = plot.setup(nrows=1)
    # fmt: off
    df.plot(
        title=f"{title}", rot=0, grid=True, use_index=True, ax=axs,
        # style=style, color=colors
    )
    # fmt: on

    plot.wrapup(pdf, show)

    # Plot each year
    for year in years:
        prox = str(int(year) + 1)
        tmp = df[df.index > dateutil.parser.parse(f"{year}-01-01")]
        tmp = tmp[tmp.index < dateutil.parser.parse(f"{prox}-01-01")]

        fig, axs = plot.setup(nrows=1)
        # fmt: off
        tmp.plot(
            title=f"{title}: {year}", rot=0, grid=True, use_index=True, ax=axs,
            # style=style, color=colors
        )
        # fmt: on

        plot.wrapup(pdf, show)


def E0(df):
    print(f"Running E0")
    with PdfPages(f"output/timeseries.pdf") as pdf:
        droplist = [column for column in df.columns if "produced-gtba" not in column]
        df = df.drop(droplist, axis=1)

        # wf = 0.5 * 2
        # hf = (5. ** 0.5 - 1.0) / 2.0 * 2
        figsize = plot.get_figsize(plot.textwidth, wf=1.0)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        chunk = df.loc[f"{2018}"]
        sns.lineplot(data=chunk, ax=ax)

        ax.get_legend().remove()
        ax.set_ylabel("Water demand (m\u00b3)")
        ax.set_xlabel("Timestamp")
        ax.yaxis.set_major_formatter(plot.OOMFormatter(3, "%1.0f"))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

        # chunk.plot(rot=90, grid=True, ax=ax, use_index=True, title=title)
        # fig.suptitle(f"{city}")

        plot.save(pdf, "timeseries")


def E1(df):
    print(f"Running E1: produced vs consumed of each city")
    # Produced vs consumed of each city
    # with PdfPages(f"output/eda/E1.pdf") as pdf:
    #     for city in ["gtba", "pp", "matinhos"]:
    #         droplist = [column for column in df.columns if city not in column]
    #         tmp = df.drop(droplist, axis=1)
    #         E1P(tmp, city, pdf, False)

    with PdfPages(f"output/produced-vs-consumed.pdf") as pdf:
        droplist = [column for column in df.columns if "gtba" not in column]
        tmp = df.drop(droplist, axis=1)
        E1P(tmp, "gtba", pdf, False)


def E2(df):
    print(f"Running E2: produced of all cities")
    # Produced of all cities
    with PdfPages(f"output/eda/E2.pdf") as pdf:
        droplist = [column for column in df.columns if "produced" not in column]
        tmp = df.drop(droplist, axis=1)
        E23P(tmp, "produced", pdf, False)


def E3(df):
    print(f"Running E3: produced of all cities")
    # Produced of all cities
    with PdfPages(f"output/eda/E3.pdf") as pdf:
        droplist = [column for column in df.columns if "consumed" not in column]
        tmp = df.drop(droplist, axis=1)
        E23P(tmp, "consumed", pdf, False)


def missingv(s):
    total = len(s)
    non = s.isna().sum()
    missing = round(non * 100 / total, 2)
    print(f"total={total}, nan={non}, notnan={total - non}, missing={missing}%")


def main():
    # @ufpr-diario: consumed in the reservoirs
    # @vp-ufpr: produced in the WTPs

    df = read("src/eda/data/raw/merged.csv")
    df = df.drop(columns=["consumed-sum-pp", "consumed-sum-gtba"])

    print(f"[consumed-gtba] Missing values (before filtering)")
    missingv(df["consumed-gtba"])

    # Filter consumed data based on produced
    consumed = df["consumed-gtba"]
    consumed = consumed[consumed <= df["produced-gtba"].max()]
    consumed = consumed[consumed >= df["produced-gtba"].min()]
    df["consumed-gtba"] = consumed

    print(f"[consumed-gtba] Missing values (after filtering)")
    missingv(df["consumed-gtba"])

    print(f"[produced-gtba] Missing values")
    missingv(df["produced-gtba"])

    print("\n-----------------------------------------------------------------------\n")

    E0(df)
    E1(df)

    # E2(df)
    # E3(df)

    # with PdfPages(f"eda/output/carnival.pdf") as pdf:
    #     # fig, axs = plt.subplots(1, 1)
    #
    #     tmp = df.loc["2017-01-01":"2017-12-31"]
    #     column = "produced-gtba"
    #
    #     axs = plt.axes()
    #     axs.plot(tmp[column])
    #
    #     start = tmp.index.get_loc("2017-03-01") - 7
    #     end = tmp.index.get_loc("2017-03-06") + 7
    #
    #     tmp = tmp.reset_index()
    #
    #     x1 = start  # tmp.index[start]
    #     x2 = end  # tmp.index[end]
    #     y1 = 15000
    #     y2 = 25000
    #     # y1 = min(tmp[column][start:end]) - np.std(tmp[column][start:end])
    #     # y2 = max(tmp[column][start:end]) + np.std(tmp[column][start:end])
    #
    #     plot.zoomin(tmp[column], (x1, x2), (y1, y2), axs)
    #     plot.wrapup(pdf, False)


if __name__ == "__main__":
    main()
