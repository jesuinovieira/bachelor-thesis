import matplotlib.backends.backend_pdf
import pandas as pd

import seaborn as sns
from scipy.spatial import distance


FILENAME = "data/dataset.csv"
LOGFILE = "output/output.log"
PDF = "output/output.pdf"


def main():
    pdf = matplotlib.backends.backend_pdf.PdfPages(PDF)
    df = pd.read_csv(FILENAME, index_col="timestamp")

    stay = [
        # "water_produced",
        "is_weekend",
        "is_holiday_curitiba",
        "is_holiday_guaratuba",
        "is_holiday_joinville",
        "is_carnival",
        "is_no_school_day",
        # "is_school_recess_pr",
    ]
    df = df.drop(columns=[col for col in df if col not in stay])
    df["is_holiday_intersection"] = (
        df.is_holiday_curitiba & df.is_holiday_guaratuba & df.is_holiday_joinville
    )
    df["is_holiday_union"] = (
        df.is_holiday_curitiba | df.is_holiday_guaratuba | df.is_holiday_joinville
    )

    import numpy as np

    dummyarray = np.empty((len(stay), len(stay)))
    dummyarray[:] = np.nan
    tmp = pd.DataFrame(dummyarray, index=stay, columns=stay)

    # Note: list of columns = list of rows
    for column in stay:
        for row in stay:
            dist = 1 - distance.jaccard(df[row], df[column])
            tmp[row][column] = dist

    sns.heatmap(tmp, fmt=".2f", cmap="YlGnBu", annot=True)
    pdf.savefig()

    pdf.close()


if __name__ == "__main__":
    main()
