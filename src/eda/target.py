import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport


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


def E1(df):
    # E1: filter consumed data
    # ----------------------------------------------------------------------------------
    # Consumed data has some astronomic values which are wrong and must be filtered

    # Pontal do parana
    df["consumed-pp-filtered"] = df["consumed-pp"][~outliers(df["consumed-pp"])]
    df["consumed-sum-pp-filtered"] = \
        df["consumed-sum-pp"][~outliers(df["consumed-sum-pp"])]

    # Guaratuba
    # Note: the filter is applied two times in order to get a decent time series
    df["consumed-gtba-filtered-1x"] = \
        df["consumed-gtba"][~outliers(df["consumed-gtba"])]
    df["consumed-gtba-filtered-2x"] = \
        df["consumed-gtba-filtered-1x"][~outliers(df["consumed-gtba-filtered-1x"])]

    df["consumed-sum-gtba-filtered-1x"] = \
        df["consumed-sum-gtba"][~outliers(df["consumed-sum-gtba"])]
    df["consumed-sum-gtba-filtered-2x"] = \
        df["consumed-sum-gtba-filtered-1x"][
            ~outliers(df["consumed-sum-gtba-filtered-1x"])
        ]

    keep = [col for col in df.columns if "consumed" in col and "-sum-" not in col]
    df.plot(y=keep, subplots=True)
    plt.show()

    df["consumed-pp"] = df["consumed-pp-filtered"]
    df["consumed-sum-pp"] = df["consumed-sum-pp-filtered"]

    df["consumed-gtba"] = df["consumed-gtba-filtered-2x"]
    df["consumed-sum-gtba"] = df["consumed-sum-gtba-filtered-2x"]

    drop = [col for col in df.columns if "filtered" in col]
    df = df.drop(columns=drop)

    keep = [col for col in df.columns if "-sum-" not in col]
    df.boxplot(column=keep)
    plt.show()

    print(df.describe)
    return df


def E2(df):
    # E2: inspect consumed and consumed-sum data
    # ----------------------------------------------------------------------------------

    keep = [col for col in df.columns if "consumed" in col]
    df.plot(y=keep, subplots=True)
    plt.show()

    print(df[keep].corr())


def main():
    sns.set_theme()
    sns.set_style("whitegrid")

    df = read("src/eda/data/raw/merged.csv")

    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    matplotlib.use(f"TkAgg")

    df = E1(df)
    E2(df)

    # https://towardsdatascience.com/exploratory-data-analysis-guide-4f9367ab05e5

    # Histogram
    # Box plot
    # Heatmap
    # Pair plot
    # Swarm plot

    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("output/eda/target.html")


if __name__ == "__main__":
    main()
