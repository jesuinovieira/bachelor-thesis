import os

import numpy as np
import pandas as pd
import requests


def readfakedata(src):
    df = pd.read_csv(src)
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")

    dst = os.path.basename(src)
    dst = dst.replace(".xlsx", ".csv")
    dst = dst.lower().replace(" ", "-")

    return df, dst


def waterproduced(src):
    # Properly ready the data
    df = pd.read_excel(src, sheet_name=None, skiprows=2, usecols=(4, 5, 6, 7, 8))
    df = df["Planilha2"]
    df = df.drop(df.tail(1).index)

    # Sistema: SAA Guaratuba
    # ETA: Todas
    df = df.drop(columns=["Sistema", "ETA", "Dia da Semana"])
    df = df.rename(columns={"Produzido (m3)": "water_produced", "Data": "timestamp"})

    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")

    dst = os.path.basename(src)
    dst = dst.replace(".xlsx", ".csv")
    dst = dst.lower().replace(" ", "-")

    return df, dst


def meteorological(src):
    df = pd.read_excel(src)

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

    # TODO: this feature engineering should be here? If inside Source, data should be
    #  already daily sampled and we wouldn't be able to create the features
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
    df = df.drop(
        columns=["temperature", "radiation", "precipitation", "relative_humidity"],
        axis=1,
    )

    df = df.dropna()

    # Reindex to fill missing dates (it will also remove data outside new idx range)
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = df.reindex(idx, fill_value=np.nan)

    dst = os.path.basename(src)
    dst = dst.replace(".xlsx", ".csv")
    dst = dst.lower().replace(" ", "-")

    return df, dst


# TODO: get from (?)
def holidays(city, state, years=("2016", "2017", "2018", "2019")):
    # Holidays data are gathered from a web API. Data from Guaratuba, Curitiba and
    # Joinville are gathered, from 2016 to 2019
    jsntoken = "amVzdWluby52ZkBnbWFpbC5jb20maGFzaD0xMDMxODQ1MTQ"

    dftmp = []
    for year in years:
        URL = (
            f"https://api.calendario.com.br/?json=true&ano="
            f"{year}&estado="
            f"{state}&cidade="
            f"{city}&token={jsntoken}"
        )

        response = requests.get(URL)
        assert response.status_code == 200, f"Reponse code {response.status_code}"

        dftmp += response.json()

    df = pd.DataFrame(dftmp)

    df = df.rename(columns={"date": "timestamp"})

    df.timestamp = pd.to_datetime(df.timestamp, format="%d/%m/%Y")
    df = df.set_index("timestamp")

    df["is_holiday_" + city.lower()] = True

    # Drop rows with holiday type = "Dia Convencional"
    df = df[df.type != "Dia Convencional"]

    # Drop not used columns
    df = df.drop(
        ["link", "description", "type_code", "raw_description", "name", "type"],
        axis=1,
    )

    # Drop duplicated rows (we're only interested if it's holiday or not)
    df = df[~df.index.duplicated(keep="first")]

    # Reindex to fill missing dates
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = df.reindex(idx, fill_value=False)

    dst = f"holidays-{city.lower()}.csv"

    return df, dst


def manual(manualist, column, dst):
    # Manual data includes carnival (with Friday, Saturday and Sunday before
    # official holidays) and school recess

    # Carnival includes Friday, Saturday and Sunday before official holidays
    # School recess for the state of Parana
    idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
    df = pd.DataFrame({"timestamp": idx})

    df = df.set_index("timestamp")  # Avoid SettingWithCopyWarning
    df[column] = False

    ixs = df.index.isin(manualist)
    df[column][ixs] = True

    return df, dst
