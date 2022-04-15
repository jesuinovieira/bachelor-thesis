import matplotlib.backends.backend_pdf
import pandas as pd

import src.lib.plot as plot


FILENAME = "data/dataset.csv"
LOGFILE = "output/output.log"
PDF = "output/output.pdf"


def main():
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDF)
    df = pd.read_csv(FILENAME, index_col="timestamp")

    df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
    df["diff"] = df.precipitation_sum_6am_to_6pm - df.precipitation_sum
    years = ["2016", "2017", "2018", "2019"]
    columns = ["precipitation_sum", "precipitation_sum_6am_to_6pm", "diff"]

    plot.yearly(df, columns, years, pdf)

    pdf.close()


if __name__ == "__main__":
    main()
