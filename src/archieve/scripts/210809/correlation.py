import matplotlib.backends.backend_pdf
import pandas as pd

import src.lib.plot as plot


FILENAME = "data/dataset.csv"
LOGFILE = "output/output.log"
PDF = "output/output.pdf"


def main():
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDF)
    df = pd.read_csv(FILENAME, index_col="timestamp")

    plot.correlation(df, "pearson", pdf)
    plot.correlation(df, "kendall", pdf)
    plot.correlation(df, "spearman", pdf)

    pdf.close()


if __name__ == "__main__":
    main()
