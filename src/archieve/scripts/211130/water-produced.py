import datetime
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import pandas as pd

import src.lib.plot as plot


DBFILE = "data/dataset.csv"
OUTPUT = "output/211130/weekly-dataset.csv"
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

    plot.wrapup(pdf=pdf)


def resample(df):
    df["water_produced"] = df.water_produced.resample("W").mean()
    df["temperature_mean"] = df.temperature_mean.resample("W").mean()
    df["radiation_mean"] = df.radiation_mean.resample("W").mean()
    df["relative_humidity_mean"] = df.relative_humidity_mean.resample("W").mean()
    df["precipitation_mean"] = df.precipitation_mean.resample("W").mean()

    # "temperature_std"
    # "radiation_std"
    # "relative_humidity_std"
    # "precipitation_std"

    df["is_holiday_ctba_gtba_jve"] = (
        df.is_holiday_ctba_gtba_jve.resample("W").sum().astype("bool")
    )
    df["is_carnival"] = df.is_carnival.resample("W").sum().astype("bool")
    df["is_school_recess_pr"] = (
        df.is_school_recess_pr.resample("W").sum().astype("bool")
    )

    df.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    with matplotlib.backends.backend_pdf.PdfPages(PDF) as pdf:
        df = pd.read_csv(DBFILE, index_col="timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.dropna()

        resample(df)

        tmp(df, 2016, 2020)
        tmp(df, 2016, 2017)
        tmp(df, 2017, 2018)
        tmp(df, 2018, 2019)
        tmp(df, 2019, 2020)
