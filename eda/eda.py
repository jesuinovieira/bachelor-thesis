import dateutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import src.plot

# Set matplotlib runtime configuration
DPI = 100
rcParams["figure.autolayout"] = True
rcParams["figure.figsize"] = (1920 / DPI, 986 / DPI)
rcParams["font.family"] = "monospace"


def read(src):
    df = pd.read_csv(src)
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")

    return df


def clean(df, cutoff):
    col = df.columns[1]
    print(f"Size: {len(df)}")

    # df = df.dropna()
    # print(f"Size: {len(df)} (drop nan)")

    df = df[df[col] > 0]
    print(f"Size: {len(df)} (drop negative values)")

    # df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    df = df[df[col] < cutoff]
    print(f"Size: {len(df)} (drop outliers, cutoff {cutoff})\n")

    return df


def everything(df, city, pdf, show):
    # Plot all years
    fig, axs = src.plot.setup(nrows=1)

    corr = df.corr().iloc[0, 1]
    title = f"{city}\ncorr={round(corr, 2)}"
    df.plot(title=title, rot=0, grid=True, use_index=True, ax=axs)
    axs.set_xlabel(None)

    src.plot.wrapup(pdf, show)


def yearly(df, city, pdf, show):
    years = [2016, 2017, 2018, 2019]
    fig, axes = plt.subplots(2, 2)

    for ax, year in zip(axes.flatten(), years):
        chunk = df.loc[f"{year}"]
        corr = chunk.corr().iloc[0, 1]

        title = f"corr={round(corr, 2)}"
        chunk.plot(rot=90, grid=True, ax=ax, use_index=True, title=title)
        ax.set_xlabel(None)

    fig.suptitle(f"{city}")
    src.plot.wrapup(pdf, show)


def monthly(df, city, pdf, show):
    years = [2016, 2017, 2018, 2019]
    for year in years:
        fig, axes = plt.subplots(2, 2)
        for month in range(0, 12):
            i = month % 4
            if i == 0 and month != 0:
                fig.suptitle(f"{city} {year}")
                src.plot.wrapup(pdf, show)
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
        src.plot.wrapup(pdf, show)


def E1P(df, city, pdf, show):
    if len(df.columns) == 1:
        return

    everything(df, city, pdf, show)
    yearly(df, city, pdf, show)
    monthly(df, city, pdf, show)


def E1(df):
    # Produced vs consumed of each city
    with PdfPages(f"eda/E1.pdf") as pdf:
        for city in ["gtba", "pp", "matinhos"]:
            droplist = [column for column in df.columns if city not in column]
            tmp = df.drop(droplist, axis=1)

            # FIXME: workaround
            tmp = clean(tmp, cutoff=30000)

            E1P(tmp, city, pdf, False)


def E23P(df, title, pdf, show):
    # TODO: move this plot to src.plot?

    # style = ["--", "--", "-", "-", "-"]
    # colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    years = [2016, 2017, 2018, 2019]

    # Plot all years
    fig, axs = src.plot.setup(nrows=1)
    # fmt: off
    df.plot(
        title=f"{title}", rot=0, grid=True, use_index=True, ax=axs,
        # style=style, color=colors
    )
    # fmt: on

    src.plot.wrapup(pdf, show)

    # Plot each year
    for year in years:
        prox = str(int(year) + 1)
        tmp = df[df.index > dateutil.parser.parse(f"{year}-01-01")]
        tmp = tmp[tmp.index < dateutil.parser.parse(f"{prox}-01-01")]

        fig, axs = src.plot.setup(nrows=1)
        # fmt: off
        tmp.plot(
            title=f"{title}: {year}", rot=0, grid=True, use_index=True, ax=axs,
            # style=style, color=colors
        )
        # fmt: on

        src.plot.wrapup(pdf, show)


def E2(df):
    # Produced of all cities
    with PdfPages(f"eda/E2.pdf") as pdf:
        droplist = [column for column in df.columns if "produced" not in column]
        tmp = df.drop(droplist, axis=1)
        E23P(tmp, "produced", pdf, False)


def E3(df):
    # Produced of all cities
    with PdfPages(f"eda/E3.pdf") as pdf:
        droplist = [column for column in df.columns if "consumed" not in column]
        tmp = df.drop(droplist, axis=1)
        E23P(tmp, "consumed", pdf, False)


def main():
    # @ufpr-diario: consumido nos reservatorios
    # @vp-ufpr: produzido nas ETAs

    df = read("eda/data/raw/merged.csv")

    # df1 = clean(df1, cutoff=30000)
    # df2 = clean(df2, cutoff=25000)

    # E0(df)
    E1(df)
    E2(df)
    E3(df)


if __name__ == "__main__":
    main()
