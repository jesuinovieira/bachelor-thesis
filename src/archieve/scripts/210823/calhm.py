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

    plot.calhm(df.is_holiday_ctba_gtba_jve, "\nHoliday GTBA CTBA JVE", pdf)
    plot.calhm(df.is_carnival, "\nCarnival", pdf)
    plot.calhm(df.is_school_recess_pr, "\nSchool Recess PR", pdf)

    pdf.close()


if __name__ == "__main__":
    main()
