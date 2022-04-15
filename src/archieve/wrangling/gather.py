"""Gather raw data and save it to 'data/raw' folder.

Note: no data processing is done in this script.
"""

import os

import pandas as pd
import requests

from src.lib import utils


FOLDER = "data/raw"
LOGFILE = "output/output.log"
logger = utils.setup_logger(__name__, LOGFILE)


def compulsory():
    """Verify compulsory files."""
    files = [
        "Guaratuba.xlsx",
        # "UFPR - DIARIO 2016 GTBA.xlsx",  # Not used
        # "UFPR - DIARIO 2017 GTBA.xlsx",  # Not used
        # "UFPR - DIARIO 2018 GTBA.xlsx",  # Not used
        # "UFPR - DIARIO 2019 GTBA.xlsx",  # Not used
        # "UFPR - HORARIO GTBA.xlsx",  # Not used
        "VP UFPR GTBA.xlsx",
    ]

    try:
        for file in files:
            path = os.path.join(FOLDER, file)
            assert os.path.isfile(path), f"Missing file '{path}'"
            logger.info(f"Found '{file}'")
    except AssertionError as err:
        logger.exception(err)


def holidays():
    """Get holidays data for the cities of Guaratuba, Curitiba and Joinville."""
    cities = [("GUARATUBA", "PR"), ("CURITIBA", "PR"), ("JOINVILLE", "SC")]
    years = ["2016", "2017", "2018", "2019"]

    for city, state in cities:
        tmp = []
        for year in years:
            URL = (
                f"https://api.calendario.com.br/?json=true&ano="
                f"{year}&estado="
                f"{state}&cidade="
                f"{city}&token=amVzdWluby52ZkBnbWFpbC5jb20maGFzaD0xMDMxODQ1MTQ"
            )

            response = requests.get(URL)
            assert response.status_code == 200, f"Reponse code {response.status_code}"

            tmp += response.json()

        df = pd.DataFrame(tmp)

        filename = f"Feriados {city}.csv"
        path = os.path.join(FOLDER, filename)

        df.to_csv(path, index=False)
        logger.info(f"Created '{path}'")


def manual():
    """Get manual data, i.e., carnival and school recess."""
    # Carnival includes Friday, Saturday and Sunday before official holidays
    # School recess for the state of Parana
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = pd.DataFrame({"timestamp": idx})
    df = df.set_index("timestamp")  # Avoid SettingWithCopyWarning

    df["is_carnival"] = False
    ixs = df.index.isin(CARNIVAL)
    df.is_carnival[ixs] = True

    df["is_school_recess_pr"] = False
    ixs = df.index.isin(SCHOOL_RECESS_PR)
    df.is_school_recess_pr[ixs] = True

    filename = "Carnival and School Recess PR (Manual).csv"
    path = os.path.join(FOLDER, filename)
    df.to_csv(path, index=True, index_label="timestamp")
    logger.info(f"Created '{path}'")


def main():
    # NOTE: reading/merging data
    run = {"compulsory": True, "holidays": True, "manual": True}

    # val = input("Files can be overwritten. Are you sure you want to continue? [y/N]")
    # if val.lower() != "y":
    #     logger.info("The operation was canceled by the user")
    #     exit(0)

    logger.info(f"Run 'compulsory()'? {run['compulsory']}")
    if run["compulsory"]:
        compulsory()

    logger.info(f"Run 'holidays()'? {run['holidays']}")
    if run["holidays"]:
        holidays()

    logger.info(f"Run 'manual()'? {run['manual']}")
    if run["manual"]:
        manual()


# ======================================================================================


# http://www.gestaoescolar.diaadia.pr.gov.br/modules/conteudo/conteudo.php?conteudo=27
CARNIVAL = (
    # http://g1.globo.com/carnaval/2016/noticia/2015/10/carnaval-2016-veja-datas.html
    pd.date_range(start="2016-02-05", end="2016-02-10").to_pydatetime().tolist()
    # https://g1.globo.com/carnaval/2017/noticia/carnaval-2017-veja-datas.ghtml
    + pd.date_range(start="2017-02-24", end="2017-03-01").to_pydatetime().tolist()
    # https://g1.globo.com/carnaval/2018/noticia/carnaval-2018-veja-datas.ghtml
    + pd.date_range(start="2018-02-09", end="2018-02-14").to_pydatetime().tolist()
    # https://g1.globo.com/carnaval/2019/noticia/carnaval-2019-veja-datas.ghtml
    + pd.date_range(start="2019-03-01", end="2019-03-06").to_pydatetime().tolist()
)

# http://www.gestaoescolar.diaadia.pr.gov.br/modules/conteudo/conteudo.php?conteudo=27
SCHOOL_RECESS_PR = (
    pd.date_range(start="2016-01-01", end="2016-02-28").to_pydatetime().tolist()
    + pd.date_range(start="2016-04-22", end="2016-04-22").to_pydatetime().tolist()
    + pd.date_range(start="2016-05-27", end="2016-05-27").to_pydatetime().tolist()
    + pd.date_range(start="2016-07-16", end="2016-07-31").to_pydatetime().tolist()
    + pd.date_range(start="2016-11-14", end="2016-11-14").to_pydatetime().tolist()
    + pd.date_range(start="2016-12-22", end="2016-12-31").to_pydatetime().tolist()
    + pd.date_range(start="2017-01-01", end="2017-02-14").to_pydatetime().tolist()
    + pd.date_range(start="2017-02-27", end="2017-02-27").to_pydatetime().tolist()
    + pd.date_range(start="2017-03-01", end="2017-03-01").to_pydatetime().tolist()
    + pd.date_range(start="2017-03-06", end="2017-03-06").to_pydatetime().tolist()
    + pd.date_range(start="2017-05-24", end="2017-05-24").to_pydatetime().tolist()
    + pd.date_range(start="2017-06-02", end="2017-06-02").to_pydatetime().tolist()
    + pd.date_range(start="2017-06-16", end="2017-06-16").to_pydatetime().tolist()
    + pd.date_range(start="2017-07-15", end="2017-07-25").to_pydatetime().tolist()
    + pd.date_range(start="2017-09-08", end="2017-09-08").to_pydatetime().tolist()
    + pd.date_range(start="2017-09-26", end="2017-09-26").to_pydatetime().tolist()
    + pd.date_range(start="2017-10-06", end="2017-10-06").to_pydatetime().tolist()
    + pd.date_range(start="2017-10-13", end="2017-10-13").to_pydatetime().tolist()
    + pd.date_range(start="2017-11-03", end="2017-11-03").to_pydatetime().tolist()
    + pd.date_range(start="2017-12-21", end="2017-12-31").to_pydatetime().tolist()
    + pd.date_range(start="2018-01-01", end="2018-02-18").to_pydatetime().tolist()
    + pd.date_range(start="2018-02-24", end="2018-02-24").to_pydatetime().tolist()
    + pd.date_range(start="2018-04-30", end="2018-04-30").to_pydatetime().tolist()
    + pd.date_range(start="2018-06-01", end="2018-06-01").to_pydatetime().tolist()
    + pd.date_range(start="2018-07-14", end="2018-07-29").to_pydatetime().tolist()
    + pd.date_range(start="2018-10-01", end="2018-10-01").to_pydatetime().tolist()
    + pd.date_range(start="2018-11-16", end="2018-11-16").to_pydatetime().tolist()
    + pd.date_range(start="2018-12-20", end="2018-12-31").to_pydatetime().tolist()
    + pd.date_range(start="2019-01-01", end="2019-02-13").to_pydatetime().tolist()
    + pd.date_range(start="2019-03-04", end="2019-03-04").to_pydatetime().tolist()
    + pd.date_range(start="2019-03-06", end="2019-03-06").to_pydatetime().tolist()
    + pd.date_range(start="2019-07-13", end="2019-07-28").to_pydatetime().tolist()
    + pd.date_range(start="2019-10-05", end="2019-10-05").to_pydatetime().tolist()
    + pd.date_range(start="2019-12-20", end="2019-12-31").to_pydatetime().tolist()
)


if __name__ == "__main__":
    main()
