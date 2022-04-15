"""Clean up raw data from 'data/raw' and save it to 'data/cache' folder.

- Create a timestamp column
- Add one object for each day from [2016-01-01, 2019-12-31]
- Remove not used columns

Note: timestamp feature engineering is also done here.
"""

import datetime
import os

import numpy as np
import pandas as pd

from src.lib import utils


FOLDER = "data/cache"
LOGFILE = "output/output.log"
logger = utils.setup_logger(__name__, LOGFILE)


def waterdemand():
    filename = "data/raw/VP UFPR GTBA.xlsx"
    df = pd.read_excel(filename, sheet_name=None, skiprows=2, usecols=(4, 5, 6, 7, 8))

    df = df["Planilha2"]
    df = df.drop(df.tail(1).index)
    df = df.drop(columns=["Sistema", "ETA", "Dia da Semana"])
    df = df.rename(columns={"Produzido (m3)": "water_produced", "Data": "timestamp"})

    path = os.path.join(FOLDER, "water-demand.csv")
    df.to_csv(path, index=False)
    logger.info(f"Cleaned up '{path}'")


def meteorological():
    filename = "data/raw/Guaratuba.xlsx"
    df = pd.read_excel(filename)
    df = df.rename(
        columns={
            "data": "timestamp",
            "temperatura": "temperature",
            "radiacao": "radiation",
            "precipitacao": "precipitation",
            "umidade_relativa": "relative_humidity",
        }
    )

    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")

    # Drop data that is not in [2016-01-01, 2019-12-31]
    start = datetime.datetime.strptime(f"2016-01-01", "%Y-%m-%d")
    end = datetime.datetime.strptime(f"2020-01-01", "%Y-%m-%d")
    df = df[(df.index >= start) & (df.index < end)]

    # Feature engineering: create daily features based on the data
    df["temperature_mean"] = df.temperature.resample("D").mean()
    df["temperature_std"] = df.temperature.resample("D").std()
    df["radiation_mean"] = df.radiation.resample("D").mean()
    df["radiation_std"] = df.radiation.resample("D").std()
    df["relative_humidity_mean"] = df.relative_humidity.resample("D").mean()
    df["relative_humidity_std"] = df.relative_humidity.resample("D").std()
    df["precipitation_mean"] = df.precipitation.resample("D").mean()
    df["precipitation_std"] = df.precipitation.resample("D").std()

    # Drop old columns
    df = df.drop(["temperature", "radiation", "precipitation", "relative_humidity"], 1)
    df = df.dropna()
    df = df.round(2)

    # Reindex to fill missing dates
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = df.reindex(idx, fill_value=np.nan)

    path = os.path.join(FOLDER, "meteorological.csv")
    df.to_csv(path, index=True, index_label="timestamp")
    logger.info(f"Cleaned up '{path}'")


def holidays():
    filenames = [
        "data/raw/Feriados CURITIBA.csv",
        "data/raw/Feriados GUARATUBA.csv",
        "data/raw/Feriados JOINVILLE.csv",
    ]

    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = pd.DataFrame({"timestamp": idx})

    # Merge holidays datasets into a single one
    for filename in filenames:
        city = filename.lstrip("data/raw/Feriados ").rstrip(".csv")
        tmp = pd.read_csv(filename)

        # Drop rows with holiday type = "Dia Convencional"
        tmp = tmp[tmp.type != "Dia Convencional"]

        # Drop not used columns
        tmp = tmp.drop(
            ["link", "description", "type_code", "raw_description", "name", "type"],
            axis=1,
        )

        tmp = tmp.rename(columns={"date": "timestamp"})
        tmp.timestamp = pd.to_datetime(tmp.timestamp, format="%d/%m/%Y")

        # Drop duplicated rows (we're only interested if it's holiday or not)
        tmp = tmp.drop_duplicates(subset="timestamp", keep="first")

        tmp["is_holiday_" + city.lower()] = True
        df = pd.merge(df, tmp, how="left", on="timestamp")

    df.timestamp = df.timestamp.dt.date
    df = df.fillna(False)

    # Create a feature that is the union of holidays in the three cities
    df["is_holiday_ctba_gtba_jve"] = (
        df.is_holiday_curitiba | df.is_holiday_guaratuba | df.is_holiday_joinville
    )
    df = df.drop(
        ["is_holiday_curitiba", "is_holiday_guaratuba", "is_holiday_joinville"], axis=1
    )

    # Add carnival special column and school recess PR
    FILENAME = "data/raw/Carnival and School Recess PR (Manual).csv"
    tmp = pd.read_csv(FILENAME)
    tmp.timestamp = pd.to_datetime(tmp.timestamp, format="%Y-%m-%d").dt.date

    df = pd.merge(df, tmp, how="left", on="timestamp")

    path = os.path.join(FOLDER, "holidays.csv")
    df.to_csv(path, index=False)
    logger.info(f"Cleaned up '{path}'")


def timestamp():
    # Datetime feature engineering
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = pd.DataFrame({"timestamp": idx})

    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d")

    if "year" not in df:
        df.insert(1, "year", df.timestamp.dt.year)
    if "month" not in df:
        df.insert(2, "month", df.timestamp.dt.month)
    if "day" not in df:
        df.insert(3, "day", df.timestamp.dt.day)
    if "dayofweek" not in df:
        df.insert(4, "dayofweek", df.timestamp.dt.dayofweek)
    if "is_weekend" not in df:
        df.insert(5, "is_weekend", np.where(df.dayofweek.isin([5, 6]), 1, 0))
    if "season" not in df:
        df.insert(6, "season", df.timestamp.apply(get_season).map(season2num))

    path = os.path.join(FOLDER, "timestamp.csv")
    df.to_csv(path, index=False)
    logger.info(f"Cleaned up '{path}'")


def main():
    run = {
        "waterdemand": True,
        "meteorological": True,
        "holidays": True,
        "timestamp": True,
    }

    # val = input("Files can be overwritten. Are you sure you want to continue? [y/N]")
    # if val.lower() != "y":
    #     logger.info("The operation was canceled by the user")
    #     exit(0)

    logger.info(f"Run 'waterdemand()'? {run['waterdemand']}")
    if run["waterdemand"]:
        waterdemand()

    logger.info(f"Run 'meteorological()'? {run['meteorological']}")
    if run["meteorological"]:
        meteorological()

    logger.info(f"Run 'holidays()'? {run['holidays']}")
    if run["holidays"]:
        holidays()

    logger.info(f"Run 'timestamp()'? {run['timestamp']}")
    if run["timestamp"]:
        timestamp()


season2num = {"Summer": 0, "Autumn": 1, "Winter": 2, "Spring": 3}

# Dicts with the end date of the seasons for each year in Brazil (UTC-3). Note that the
# date is the last one of that season: for example, summer goes until '2016-03-19',
# '2016-03-20' is already Autumn
S2016 = {
    "Summer": datetime.datetime.strptime("2016-03-20", "%Y-%m-%d"),
    "Autumn": datetime.datetime.strptime("2016-06-20", "%Y-%m-%d"),
    "Winter": datetime.datetime.strptime("2016-09-22", "%Y-%m-%d"),
    "Spring": datetime.datetime.strptime("2016-12-21", "%Y-%m-%d"),
}

S2017 = {
    "Summer": datetime.datetime.strptime("2017-03-20", "%Y-%m-%d"),
    "Autumn": datetime.datetime.strptime("2017-06-21", "%Y-%m-%d"),
    "Winter": datetime.datetime.strptime("2017-09-22", "%Y-%m-%d"),
    "Spring": datetime.datetime.strptime("2017-12-21", "%Y-%m-%d"),
}

S2018 = {
    "Summer": datetime.datetime.strptime("2018-03-20", "%Y-%m-%d"),
    "Autumn": datetime.datetime.strptime("2018-06-21", "%Y-%m-%d"),
    "Winter": datetime.datetime.strptime("2018-09-23", "%Y-%m-%d"),
    "Spring": datetime.datetime.strptime("2018-12-21", "%Y-%m-%d"),
}

S2019 = {
    "Summer": datetime.datetime.strptime("2019-03-20", "%Y-%m-%d"),
    "Autumn": datetime.datetime.strptime("2019-06-21", "%Y-%m-%d"),
    "Winter": datetime.datetime.strptime("2019-09-23", "%Y-%m-%d"),
    "Spring": datetime.datetime.strptime("2019-12-22", "%Y-%m-%d"),
}


def get_season(date):
    # Return season the date falls in.
    if date.year == 2016:
        S = S2016
    if date.year == 2017:
        S = S2017
    if date.year == 2018:
        S = S2018
    if date.year == 2019:
        S = S2019

    if date < S["Summer"]:
        return "Summer"
    if date < S["Autumn"]:
        return "Autumn"
    if date < S["Winter"]:
        return "Winter"
    if date < S["Spring"]:
        return "Spring"

    return "Summer"


if __name__ == "__main__":
    main()
