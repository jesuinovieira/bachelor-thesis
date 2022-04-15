import datetime
import os
import shutil

import numpy as np
import pandas as pd
import requests

import src.utils as utils
from src._manual import CARNIVAL
from src._manual import SCHOOL_RECESS_PR


class Source:
    def __init__(self, raw, preprocessed, output):
        if not os.path.isdir(raw):
            os.makedirs(raw)
        if not os.path.isdir(preprocessed):
            os.makedirs(preprocessed)
        if not os.path.isdir(output):
            os.makedirs(output)

        self.raw = raw
        self.preprocessed = preprocessed
        self.output = output

    def select(self):
        """Gather data and save it in 'data/raw'."""
        self._gather_compulsory()
        self._gather_holidays()
        self._gather_manual()

    def _gather_compulsory(self):
        # Compulsory or "FromDirectory", data that is already in the filesystem
        # TODO: hardcoded filepaths
        filepaths = ["data/external/Guaratuba.xlsx", "data/external/VP UFPR GTBA.xlsx"]

        try:
            for src in filepaths:
                assert os.path.isfile(src), f"Missing file '{src}'"

                basename = os.path.basename(src)
                basename = basename.lower().replace(" ", "-")
                dst = os.path.join(self.raw, basename)
                shutil.copy(src, dst)

                print(f"Gathered '{dst}' file")
        except AssertionError as err:
            print(err)

    def _gather_holidays(self):
        # Holidays data are gathered from a web API. Data from Guaratuba, Curitiba and
        # Joinville are gathered, from 2016 to 2019
        # TODO: hardcoded URLs
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
                assert (
                    response.status_code == 200
                ), f"Reponse code {response.status_code}"

                tmp += response.json()

            df = pd.DataFrame(tmp)

            filename = f"feriados-{city.lower()}.csv"
            dst = os.path.join(self.raw, filename)
            df.to_csv(dst, index=False)
            print(f"Gathered '{dst}' file")

    def _gather_manual(self):
        # Manual data includes carnival (with Friday, Saturday and Sunday before
        # official holidays) and school recess

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

        filename = "carnival-and-school-recess-parana.csv"
        dst = os.path.join(self.raw, filename)
        df.to_csv(dst, index=True, index_label="timestamp")
        print(f"Gathered '{dst}' file")

    def preprocess(self):
        # val = input(
        #     "Files can be overwritten. Are you sure you want to continue? [y/N]"
        # )
        # if val.lower() != "y":
        #     print("The operation was canceled by the user")
        #     exit(0)

        self._preprocess_waterdemand()
        self._preprocess_meteorological()
        self._preprocess_holidays()

        # Feature engineering
        self._timestamp()

        # Publish: publish.py
        idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
        df = pd.DataFrame({"timestamp": idx})
        df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d").dt.date

        for file in os.listdir(self.preprocessed):
            path = os.path.join(self.preprocessed, file)

            assert path.endswith(".csv"), f"File {path} doesn't end with '.csv'"
            assert os.path.isfile(path), f"Not a file '{path}'"

            tmp = pd.read_csv(path)
            tmp.timestamp = pd.to_datetime(tmp.timestamp, format="%Y-%m-%d").dt.date

            assert utils.validate(tmp), f"File '{file}' didn't pass validate test"

            print(f"Merging '{file}'")
            df = pd.merge(df, tmp, how="left", on="timestamp")

        assert utils.validate(df), f"Final dataframe didn't pass validate test"

        df = df.dropna()
        df = df.set_index("timestamp")

        # ==============================================================================
        # TODO: improve the way (and also where atributes are ordered).. maybe it makes
        #  more sense to name x1, ..., xn and y?
        attributes = utils.fs2attributes["FS1"]
        df = df.drop(columns=[col for col in df if col not in attributes])
        df = df[attributes]
        # ==============================================================================

        dst = os.path.join(self.output, "dataset.csv")
        df.to_csv(dst)
        print(f"Final dataframe '{dst}' is ready :)")

        return df

    def _preprocess_waterdemand(self):
        filename = os.path.join(self.raw, "vp-ufpr-gtba.xlsx")
        df = pd.read_excel(
            filename, sheet_name=None, skiprows=2, usecols=(4, 5, 6, 7, 8)
        )

        df = df["Planilha2"]
        df = df.drop(df.tail(1).index)
        df = df.drop(columns=["Sistema", "ETA", "Dia da Semana"])
        df = df.rename(
            columns={"Produzido (m3)": "water_produced", "Data": "timestamp"}
        )

        dst = os.path.join(self.preprocessed, "water-demand.csv")
        df.to_csv(dst, index=False)
        print(f"Cleaned up '{dst}'")

    def _preprocess_meteorological(self):
        filename = os.path.join(self.raw, "guaratuba.xlsx")
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

        # TODO: modularize feature engineering step? IOW, remove it from here?
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
        df = df.round(2)

        # Reindex to fill missing dates
        idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
        df = df.reindex(idx, fill_value=np.nan)

        dst = os.path.join(self.preprocessed, "meteorological.csv")
        df.to_csv(dst, index=True, index_label="timestamp")
        print(f"Cleaned up '{dst}'")

    def _preprocess_holidays(self):
        filenames = [
            os.path.join(self.raw, "feriados-curitiba.csv"),
            os.path.join(self.raw, "feriados-guaratuba.csv"),
            os.path.join(self.raw, "feriados-joinville.csv"),
        ]

        idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
        df = pd.DataFrame({"timestamp": idx})

        # Merge holidays datasets into a single dataframe
        for filename in filenames:
            string = os.path.join(self.raw, "feriados-")
            city = filename.lstrip(string).rstrip(".csv")

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
            ["is_holiday_curitiba", "is_holiday_guaratuba", "is_holiday_joinville"],
            axis=1,
        )

        # Add carnival special column and school recess PR
        FILENAME = os.path.join(self.raw, "carnival-and-school-recess-parana.csv")
        tmp = pd.read_csv(FILENAME)
        tmp.timestamp = pd.to_datetime(tmp.timestamp, format="%Y-%m-%d").dt.date

        df = pd.merge(df, tmp, how="left", on="timestamp")

        dst = os.path.join(self.preprocessed, "holidays.csv")
        df.to_csv(dst, index=False)
        print(f"Cleaned up '{dst}'")

    def _timestamp(self):
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
            df.insert(
                6, "season", df.timestamp.apply(utils.get_season).map(utils.season2num)
            )

        dst = os.path.join(self.preprocessed, "timestamp.csv")
        df.to_csv(dst, index=False)
        print(f"Cleaned up '{dst}'")
