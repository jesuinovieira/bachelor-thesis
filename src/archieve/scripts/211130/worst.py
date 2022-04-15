import datetime
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd

import src.lib.plot as plot


DBFILE = "data/dataset.csv"
PDF = "output/output.pdf"


def tmp(df, start, end):
    start = datetime.datetime.strptime(f"{start}-01-01", "%Y-%m-%d")
    end = datetime.datetime.strptime(f"{end}-01-01", "%Y-%m-%d")
    df = df[(df.index >= start) & (df.index < end)]

    # Plot all data
    df.water_produced.plot(c="tab:blue", linewidth=2)

    # Sample weekly
    df["water_produced"] = df.water_produced.resample("W").mean()
    df = df.dropna()

    # Plot again
    df.water_produced.plot(c="tab:orange", linewidth=2)

    # plt.show()
    plot.wrapup(pdf=pdf)


if __name__ == "__main__":
    with matplotlib.backends.backend_pdf.PdfPages(PDF) as pdf:
        df = pd.read_csv(DBFILE, index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.dropna()

        tmp(df, 2016, 2020)
        tmp(df, 2016, 2017)
        tmp(df, 2017, 2018)
        tmp(df, 2018, 2019)
        tmp(df, 2019, 2020)
