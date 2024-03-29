import os

import numpy as np
import pandas as pd

import src.utils as utils


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

    def select(self, callback):
        """Gather data and save it in `self.raw` folder."""
        df, dst = callback()
        dst = os.path.join(self.raw, dst)
        df.to_csv(dst, index=True, index_label="timestamp")
        print(f"Gathered '{dst}' file")

    def preprocess(self):
        df = self._merge()
        df = self._clean(df)
        df = self._fe(df)

        dst = os.path.join(self.output, "dataset.csv")
        df.to_csv(dst)
        print(f"Final dataframe '{dst}' is ready :)")

        return df

    def getdst(self):
        return os.path.join(self.output, "dataset.csv")

    def _merge(self):
        # NOTE: data outside this range will be filtered out
        idx = pd.date_range("2016-01-01", "2019-12-31", freq="D")
        df = pd.DataFrame({"timestamp": idx})
        df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")

        for file in os.listdir(self.raw):
            src = os.path.join(self.raw, file)

            tmp = pd.read_csv(src)
            tmp.timestamp = pd.to_datetime(tmp.timestamp, format="%Y-%m-%d")

            df = pd.merge(df, tmp, how="left", on="timestamp")
            print(f"Merging '{file}'")

        df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("timestamp")

        return df

    @staticmethod
    def _clean(df):
        # Ideas:
        #
        # [ ] Make it configurable?
        # [ ] Remove noise
        # [ ] Remove outliers
        # [ ] Handle missing values
        # [ ] Onehotencoding (?) this is not cleaning, but might be... what?

        df = df.dropna()  # Drop rows with any missing value
        df = df.round(2)  # Round numeric columns

        return df

    @staticmethod
    def _fe(df):
        # Datetime feature engineering
        if "year" not in df:
            df.insert(0, "year", df.index.year)
        if "month" not in df:
            df.insert(1, "month", df.index.month)
        if "day" not in df:
            df.insert(2, "day", df.index.day)
        if "dayofweek" not in df:
            df.insert(3, "dayofweek", df.index.dayofweek)
        if "is_weekend" not in df:
            df.insert(4, "is_weekend", np.where(df.index.dayofweek.isin([5, 6]), 1, 0))
        if "season" not in df:
            df.insert(
                5,
                "season",
                df.index.to_series().apply(utils.get_season).map(utils.season2num),
            )

        # FIXME: hardcoded
        # Create a feature that is the union of holidays in the three cities
        df["is_holiday_ctba_gtba_jve"] = (
            df.is_holiday_curitiba | df.is_holiday_guaratuba | df.is_holiday_joinville
        )
        df = df.drop(
            ["is_holiday_curitiba", "is_holiday_guaratuba", "is_holiday_joinville"],
            axis=1,
        )

        return df
