import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import src.utils

# TODO: orchestrate model training (googleit)


class Processor:
    def __init__(self, name, df, dbname, vm, trainw, n_splits):
        self._name = name

        self.df = df
        self.model = None
        self._scaler = MinMaxScaler()

        self.vm = vm
        self.trainw = trainw
        self.n_splits = n_splits
        self.pr = ProcessorResults(
            self._name, self.model, self.vm, self.trainw, self.n_splits, dbname
        )

        # Order columns in dataframe
        attr = src.utils.fs2attributes[dbname]
        self.df = self.df[attr]

        # Initialize cross validator object
        length = len(self.df.index)

        # from sklearn.model_selection import TimeSeriesSplit
        # self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        vm = src.utils.ExpandingWindow if self.vm == "EW" else src.utils.SlidingWindow
        self.tscv = vm(length, self.n_splits, self.trainw)

    def transform(self):
        self._scaler.fit(self.df)
        self.df[self.df.columns] = self._scaler.transform(self.df)

    def fit(self):
        X, y = src.utils.df2np(self.df)

        # Cross-validation
        for i, (trainidxs, testidxs) in enumerate(self.tscv.split(X, y)):
            X_train, X_test = X[trainidxs], X[testidxs]
            y_train, y_test = y[trainidxs], y[testidxs]

            self.model.fit(X_train, y_train)
            y_hat = self.predict(X_test)

            # Rescale the target and save results
            self.pr.add(
                yhat=src.utils.rescaletarget(self._scaler, y_hat),
                ytrue=src.utils.rescaletarget(self._scaler, y_test),
            )

        timestamps = self.df.index[self.trainw :].to_numpy()
        self.pr.save(timestamps)

        return self.pr

    def predict(self, X):
        return self.model.predict(X)


class ProcessorResults:
    def __init__(
        self, name, model, vm, trainw, n_splits, dbname, path="output/processor"
    ):
        self.id = f"{dbname}-{name}-{vm}-{trainw}-{n_splits}"
        self.model = model
        self.output = path

        self.vm = vm  # Validation method
        self.trainw = trainw  # Train window
        self.n_splits = n_splits  # Number of splits for cross validation
        self.dbname = dbname  # Database name

        self.yhat = np.array([], dtype=float)
        self.ytrue = np.array([], dtype=float)
        self.timestamp = np.array([], dtype=float)

    def add(self, yhat, ytrue):
        assert yhat.size == ytrue.size
        self.yhat = np.append(self.yhat, yhat)
        self.ytrue = np.append(self.ytrue, ytrue)

    def save(self, timestamps):
        if not os.path.isdir(self.output):
            os.makedirs(self.output, exist_ok=True)

        if not self.timestamp.size:
            self.timestamp = timestamps

        assert self.yhat.size == self.ytrue.size == self.timestamp.size

        # Save predictions
        data = {"ytrue": self.ytrue, "yhat": self.yhat}
        df = pd.DataFrame(data=data, index=timestamps)
        dst = os.path.join(self.output, f"{self.id}.csv")
        df.to_csv(dst, index=True, index_label="timestamp")

        # Save model
        dst = os.path.join(self.output, f"{self.id}.pickle")
        with open(dst, "wb") as f:
            pickle.dump(self, f)

    def getdst(self):
        return os.path.join(self.output, f"{self.id}.pickle")


class LRProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, n_splits):
        super().__init__(id, df, dbname, vm, trainw, n_splits)
        self.model = LinearRegression()


class KNNProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, n_splits):
        super().__init__(id, df, dbname, vm, trainw, n_splits)
        self.model = KNeighborsRegressor(
            n_neighbors=11, weights="distance", metric="minkowski", p=2
        )


class SVRProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, n_splits):
        super().__init__(id, df, dbname, vm, trainw, n_splits)
        self.model = SVR(kernel="rbf", C=1, epsilon=0.1)


class MLPProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, n_splits):
        super().__init__(id, df, dbname, vm, trainw, n_splits)
        self.model = MLPRegressor(
            hidden_layer_sizes=(17, ),
            activation="relu",
            learning_rate_init=0.01,
            max_iter=1000,
            tol=1e-4,
            momentum=0.9,
            early_stopping=True,
            random_state=1,
        )
