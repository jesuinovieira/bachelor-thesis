import matplotlib.backends.backend_pdf
import pandas as pd

import src.lib.plot as plot


FILENAME = "data/dataset.csv"
LOGFILE = "output/output.log"
PDF = "output/output.pdf"


def main():
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDF)
    df = pd.read_csv(FILENAME, index_col="timestamp")

    stay = ["water_produced", "precipitation_sum", "precipitation_sum_6am_to_6pm"]
    df = df.drop(columns=[col for col in df if col not in stay])

    df["precipitation_occ_higher_than_mean"] = (
        df.precipitation_sum > df.precipitation_sum.mean()
    )
    df["precipitation_occ_higher_than_median"] = (
        df.precipitation_sum > df.precipitation_sum.median()
    )
    df["precipitation_occ_higher_than_mean_6am_to_6pm"] = (
        df.precipitation_sum_6am_to_6pm > df.precipitation_sum_6am_to_6pm.mean()
    )
    df["precipitation_occ_higher_than_median_6am_to_6pm"] = (
        df.precipitation_sum_6am_to_6pm > df.precipitation_sum_6am_to_6pm.median()
    )

    plot.correlation(df, "pearson", pdf)
    plot.correlation(df, "kendall", pdf)
    plot.correlation(df, "spearman", pdf)

    pdf.close()


if __name__ == "__main__":
    main()
