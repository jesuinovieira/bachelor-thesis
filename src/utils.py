import datetime

import numpy as np
import pandas as pd

season2num = {"Summer": 0, "Autumn": 1, "Winter": 2, "Spring": 3}

# NOTE: Dicts with the end date of the seasons for each year in Brazil (UTC-3). The date
#  is the last one of that season: for example, summer goes until '2016-03-19',
#  '2016-03-20' is already Autumn
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

TARGET = ["water_produced"]
METEOROLOGICAL = [
    "temperature_mean",
    "temperature_std",
    "radiation_mean",
    "radiation_std",
    "relative_humidity_mean",
    "relative_humidity_std",
    "precipitation_mean",
    "precipitation_std",
]
TIMESTAMP = ["year", "month", "day", "dayofweek", "is_weekend", "season"]
HOLIDAYS = ["is_holiday_ctba_gtba_jve", "is_carnival", "is_school_recess_pr"]

fs2attributes = {
    "FS1": TARGET + METEOROLOGICAL + TIMESTAMP + HOLIDAYS,
    "FS2": TARGET + METEOROLOGICAL + TIMESTAMP,
    "FS3": TARGET + HOLIDAYS + TIMESTAMP,
    "FS4": TARGET + TIMESTAMP,
}


def get_season(date):
    # Return season the date falls in
    assert 2016 <= date.year <= 2019

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


def validate(dataframe):
    # dataframe = dataframe.set_index("timestamp")
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
        print(err)
        return False

    return True


def cvsplit(trainw, testw, vm, df):
    n = len(df)
    data = df.to_numpy()

    for i in range(trainw, n, testw):
        trainidxs = slice(i - trainw if vm == "SW" else 0, i)
        testidxs = slice(i, i + testw)

        # print(f"Train: {trainidxs}")
        # print(f"Test: {testidxs}")

        # Train and test split
        train = data[trainidxs]
        test = data[testidxs]

        # Rescale the data
        # scaler = MinMaxScaler()
        # scaler.fit(train)
        # train = scaler.transform(train)
        # test = scaler.transform(test)

        # Split input (X) and output (y) vectors
        X_train, y_train = train[:, 1:], train[:, 0].T
        X_test, y_test = test[:, 1:], test[:, 0].T

        yield X_train, X_test, y_train, y_test


def rescaletarget(scaler, y):
    # Scaler was build with a dataframe, and now we want to rescale only the target
    n = scaler.n_features_in_ - 1

    length = len(y)
    X = np.zeros((length, n))
    y = y.reshape(-1, 1)
    data = np.append(y, X, axis=1)
    y = scaler.inverse_transform(data)[:, 0].T

    return y
