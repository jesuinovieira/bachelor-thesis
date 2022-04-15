"""Validade and merge all datasets from 'data/cache' into a single one which is
publishedin 'OUTPUT' file.

Note: datetime feature engineering is also done here, since it's better to do it in the
already merged database.
"""

import datetime
import os

import pandas as pd

from src.lib import utils


FOLDER = "data/cache"
OUTPUT = "data/dataset.csv"
LOGFILE = "output/output.log"
logger = utils.setup_logger(__name__, LOGFILE)


def validate(dataframe):
    dataframe = dataframe.set_index("timestamp")
    dates = pd.date_range(start="1/1/2016", end="12/31/2019")

    try:
        # Assert that there are no duplicated rows
        assert any(dataframe.index.duplicated()) is False, f"Found duplicated rows"
        # Assert that df has 1561 rows, number of days from [2016-01-01, 2019-12-31]
        assert len(dates) == len(dataframe), "Dataframe must have 1461 rows"

        # Assert that there is one row for each date from [2016-01-01, 2019-12-31]
        for date in dates:
            date = datetime.datetime.date(date)
            try:
                dataframe.loc[date]
            except KeyError as err:
                raise AssertionError(err)

    except AssertionError as err:
        logger.error(err)
        return False

    return True


def main():
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = pd.DataFrame({"timestamp": idx})
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d").dt.date

    for file in os.listdir(FOLDER):
        path = os.path.join(FOLDER, file)

        assert path.endswith(".csv"), f"File {path} doesn't end with '.csv'"
        assert os.path.isfile(path), f"Not a file '{path}'"

        tmp = pd.read_csv(path)
        tmp.timestamp = pd.to_datetime(tmp.timestamp, format="%Y-%m-%d").dt.date

        assert validate(tmp), f"File '{file}' didn't pass validate test"

        logger.info(f"Merging '{file}'")
        df = pd.merge(df, tmp, how="left", on="timestamp")

    assert validate(df), f"Final dataframe didn't pass validate test"
    logger.info(f"Final dataframe '{OUTPUT}' is ready :)")
    df.to_csv(OUTPUT, index=False)


if __name__ == "__main__":
    main()
