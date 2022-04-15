import dateutil.parser

import pandas as pd
import matplotlib.backends.backend_pdf
from matplotlib import rcParams

from src import plot
from src import utils


# Volumes de água consumidos por hora nos reservatórios!
FILENAME = "data/raw/UFPR - HORARIO GTBA.xlsx"
OUTPUT = "output/output.pdf"
LOGFILE = "output/output.log"

# Set matplotlib runtime configuration
DPI = 100
rcParams["figure.autolayout"] = True
rcParams["figure.figsize"] = (1920 / DPI, 986 / DPI)
rcParams["font.family"] = "monospace"

logger = utils.setup_logger(__name__, LOGFILE)


if __name__ == "__main__":
    districts = ["Coroados", "Central", "Brejatuba", "Aeroporto"]
    pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT)

    # Read four sheets: one for each year
    dfs = pd.read_excel(FILENAME, sheet_name=None, skiprows=2, usecols=(3, 4, 5, 6, 7))

    # Concat them to create a unique dataframe
    df = pd.concat(dfs.values())
    del dfs

    df = df.rename(
        columns={
            "Unnamed: 3": "Timestamp",
            "COROADOS": "Coroados",
            "CENTRAL": "Central",
            "BREJATUBA": "Brejatuba",
            "AEROPORTO": "Aeroporto",
        }
    )
    df = df.set_index("Timestamp")

    # ----------------------------------------------------------------------------------

    distr = "Aeroporto"

    # TODO: work with Nans and complete df instead of removing coluns
    if distr == "Coroados":
        df = df.drop(["Central", "Brejatuba", "Aeroporto"], axis=1)
    if distr == "Central":
        df = df.drop(["Coroados", "Brejatuba", "Aeroporto"], axis=1)
    if distr == "Brejatuba":
        df = df.drop(["Coroados", "Central", "Aeroporto"], axis=1)
    if distr == "Aeroporto":
        df = df.drop(["Coroados", "Central", "Brejatuba"], axis=1)

    length = len(df.index)

    # Drop rows with 'Bad' values
    # df = df[~df.eq("Bad")]
    df = df[~df[distr].eq("Bad")]

    badrows = length - len(df.index)
    length = len(df.index)
    logger.info(f"Number of rows with value 'Bad': {badrows}")

    # Drop rows with initial value <= 0
    # Since we will use diff(), the previous value must be the accumulated one
    df = df[~df[distr].le(0)]

    badrows = length - len(df.index)
    logger.info(f"Number of rows with value lower than or equal '0': {badrows}")

    # Discrete difference of each row
    df[distr] = df[distr].diff()

    # Daily downsampling and sum values
    # TODO: record at 00:00 should be in the previous day?
    df = df.resample("D").sum()
    df[distr] = df[distr].astype(int)

    length = len(df.index)

    # Drop rows where the value after downsampling is <= 0
    # Function resample add all missing days filled by NaNs
    # df = df.dropna(how="all")
    df = df[~df[distr].le(0)]

    badrows = length - len(df.index)
    logger.info(
        f"Number of rows with value lower than or equal '0' after diff: {badrows}"
    )

    # Add weekday column
    df.insert(1, "Weekday", df.index.to_series().dt.day_name())

    # ----------------------------------------------------------------------------------

    # TODO:
    #  Plot Y axis units (m³?)

    # Only for better visualization
    # df = df[~df[distr].gt(10000)]

    last = None
    years = ["2016", "2017", "2018", "2019"]

    # Plot separate annual data
    for year in years:
        prox = str(int(year) + 1)
        tmp = df[df.index > dateutil.parser.parse(f"{year}-01-01")]
        tmp = tmp[tmp.index < dateutil.parser.parse(f"{prox}-01-01")]

        _, axs = plot.setup(nrows=1)
        tmp.plot(
            kind="line",
            style=".-",
            title=f"{year}",
            y=distr,
            use_index=True,
            ax=axs,
        )
        plot.wrapup(pdf, False)

    # Plot all data
    _, axs = plot.setup(nrows=1)
    df.plot(kind="line", style=".-", y=distr, use_index=True, ax=axs)
    plot.wrapup(pdf, False)

    pd.set_option("display.float_format", lambda x: "%.5f" % x)
    logger.debug(df.describe())

    pdf.close()
