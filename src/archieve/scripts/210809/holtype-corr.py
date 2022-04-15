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
        "is_holiday_curitiba",
        "is_holiday_guaratuba",
        "is_holiday_joinville",
    ]
    df = df.drop(columns=[col for col in df if col not in stay])

    plot.correlation(df, "pearson", pdf)

    pdf.close()


if __name__ == "__main__":
    main()
