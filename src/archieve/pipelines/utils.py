import copy

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


DBFILE = "data/dataset.csv"
# DBFILE = "output/211130/weekly-dataset.csv"


def getdata(trainw, testw, vm, subset):
    assert vm == "EW" or vm == "SW"

    df = pd.read_csv(DBFILE)
    df = getsubset(df, subset)
    df = df.dropna()

    n = len(df)
    data = df.to_numpy()
    # columns = df.columns

    # TODO: probably it's wrong to fit scaler in the complete data, but I'm also not
    #  sure if it's correct to scale after "cross validation"
    # https://stackoverflow.com/questions/49444262/normalize-data-before-or-after-split-of-training-and-testing-data
    # https://towardsdatascience.com/preventing-data-leakage-in-your-machine-learning-model-9ae54b3cd1fb
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    scaler = MinMaxScaler()
    scaler.fit(df)
    df[df.columns] = scaler.transform(df)

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

        yield X_train, X_test, y_train, y_test, scaler


def normalize():
    # TODO: probably it's wrong to fit scaler in the complete data, but I'm also not
    #  sure if it's correct to scale after "cross validation"
    # https://stackoverflow.com/questions/49444262/normalize-data-before-or-after-split-of-training-and-testing-data
    # https://towardsdatascience.com/preventing-data-leakage-in-your-machine-learning-model-9ae54b3cd1fb
    # scaler = MinMaxScaler()
    # scaler.fit(data)
    # data = scaler.transform(data)

    # Rescale the data
    # scaler = MinMaxScaler()
    # scaler.fit(train)
    pass


def getsubset(df, subset):
    # Subsets
    # DS1: complete database
    # DS2: meteorological attributes
    # DS3: attributes derived from timestamp
    # DS4: holidays and school recess attributes
    attributes = fs2attributes[subset]

    df = df.drop(columns=[col for col in df if col not in attributes])
    df = df[attributes]

    return df


def rescale(scaler, y):
    n = scaler.n_features_in_ - 1

    length = len(y)
    X = np.zeros((length, n))
    y = y.reshape(-1, 1)
    data = np.append(y, X, axis=1)
    y = scaler.inverse_transform(data)[:, 0].T

    return y


fs2attributes = {
    "FS1": [
        "water_produced",
        "temperature_mean",
        "temperature_std",
        "radiation_mean",
        "radiation_std",
        "relative_humidity_mean",
        "relative_humidity_std",
        "precipitation_mean",
        "precipitation_std",
        "year",
        "month",
        "day",
        "dayofweek",
        "is_weekend",
        "season",
        "is_holiday_ctba_gtba_jve",
        "is_carnival",
        "is_school_recess_pr",
    ],
    "FS2": [
        "water_produced",
        "temperature_mean",
        "temperature_std",
        "radiation_mean",
        "radiation_std",
        "relative_humidity_mean",
        "relative_humidity_std",
        "precipitation_mean",
        "precipitation_std",
        "year",
        "month",
        "day",
        "dayofweek",
        "is_weekend",
        "season",
    ],
    "FS3": [
        "water_produced",
        "is_holiday_ctba_gtba_jve",
        "is_carnival",
        "is_school_recess_pr",
        "year",
        "month",
        "day",
        "dayofweek",
        "is_weekend",
        "season",
    ],
    "FS4": [
        "water_produced",
        "year",
        "month",
        "day",
        "dayofweek",
        "is_weekend",
        "season",
    ],
}
