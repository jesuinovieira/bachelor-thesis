import matplotlib.backends.backend_pdf
import pandas as pd

import src.lib.plot as plot


FILENAME = "data/dataset.csv"
LOGFILE = "output/output.log"
PDF = "output/output.pdf"


def main():
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDF)
    df = pd.read_csv(FILENAME, index_col="timestamp")

    stay = [
        "water_produced",
        # "temperature_max",
        # "temperature_min",
        # "temperature_avg",
        # "radiation_avg",
        # "relative_humidity_avg",
        # "precipitation_sum_6am_to_6pm",
        # "precipitation_sum",
        "temperature_mean",
        "temperature_std",
        "radiation_mean",
        "radiation_std",
        "relative_humidity_mean",
        "relative_humidity_std",
        "precipitation_mean",
        "precipitation_std",
        "year",
        # "month",
        # "day",
        # "dayofweek",
        # "season",
    ]
    df1 = df.drop(columns=[col for col in df if col not in stay])

    stay = [
        "water_produced",
        "is_weekend",
        "is_holiday_curitiba",
        "is_holiday_guaratuba",
        "is_holiday_joinville",
        "is_carnival",
        "is_school_recess_pr",
        # "is_no_school_day",
    ]
    df2 = df.drop(columns=[col for col in df if col not in stay])
    df2["is_holiday_intersection"] = (
        df2.is_holiday_curitiba & df2.is_holiday_guaratuba & df2.is_holiday_joinville
    )
    df2["is_holiday_union"] = (
        df2.is_holiday_curitiba | df2.is_holiday_guaratuba | df2.is_holiday_joinville
    )

    plot.correlation(df1, "pearson", pdf)
    plot.pbc(df2, "water_produced", pdf)
    # plot.correlation(df2, "pearson", pdf)

    pdf.close()


if __name__ == "__main__":
    main()
