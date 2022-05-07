import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import src.utils
import src.tscv as tscv

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, name, df, vm, trainw, n_splits, output):
        self._name = name
        self.method = None
        self.model = None
        self.space = None
        self.defaults = None

        self.df = df
        self._scaler = MinMaxScaler()

        self.vm = vm
        self.trainw = trainw
        self.n_splits = n_splits
        self.pr = ProcessorResults(
            self._name, self.model, self.vm, self.trainw, self.n_splits, path=output
        )

        # TODO: choose based on self.vm
        # from sklearn.model_selection import TimeSeriesSplit
        # self.tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.tscv = tscv.ExpandingWindow if self.vm == "EW" else tscv.SlidingWindow

    def transform(self):
        self._scaler.fit(self.df)
        self.df[self.df.columns] = self._scaler.transform(self.df)

    def oldfit(self):
        X, y = src.utils.df2np(self.df)

        # Grid search with time series cross validation
        n_jobs = -1
        scoring = "neg_root_mean_squared_error"
        rows, _ = X.shape
        self.tscv = self.tscv(rows, self.n_splits, self.trainw)

        search = GridSearchCV(
            self.model, self.space, cv=self.tscv, scoring=scoring, n_jobs=n_jobs
        )
        result = search.fit(X, y)

        # Log and save outputs
        logger.info(f"Best params for '{self._name}': {result.best_params_}")

        folds, ytrue, yhat, timestamps = self.crossvalidate(X, y, result)
        self.pr.add(folds, ytrue, yhat, timestamps)

        rmse = round(mean_squared_error(self.pr.ytrue, self.pr.yhat, squared=False), 3)
        mape = round(mean_absolute_percentage_error(self.pr.ytrue, self.pr.yhat), 3)
        r2 = round(r2_score(self.pr.ytrue, self.pr.yhat), 3)
        logger.info(f"[val/test] RMSE={rmse}, MAPE={mape:.3f}, R2={r2:.3f}")

        return self.pr

    def fit(self):
        # Split data into train and test set (validation set is included in train)
        # TODO: train_test_split by timestamp
        train, test = train_test_split(self.df, test_size=0.2475, shuffle=False)

        data = train.to_numpy()
        X_train, y_train = data[:, 1:], data[:, 0].T

        data = test.to_numpy()
        X_test, y_test = data[:, 1:], data[:, 0].T

        # Model selection
        n_jobs = -1
        scoring = "neg_root_mean_squared_error"
        rows, _ = X_train.shape
        self.tscv = self.tscv(rows, self.n_splits, self.trainw)

        search = GridSearchCV(
            self.model, self.space, cv=self.tscv, scoring=scoring, n_jobs=n_jobs
        )
        result = search.fit(X_train, y_train)

        # Fit selected model in train data
        model = self.method(**self.defaults, **result.best_params_)
        model.fit(X_train, y_train)

        # Evaluate on test data
        y_hat = model.predict(X_test)

        self.pr.add(
            split=np.full(shape=y_hat.size, fill_value=0),
            yhat=src.utils.rescaletarget(self._scaler, y_hat),
            ytrue=src.utils.rescaletarget(self._scaler, y_test),
            timestamp=test.index.to_numpy()
        )

        logger.info(f"Best params for '{self._name}': {result.best_params_}")

        _, _ytrue, _yhat, _ = self.crossvalidate(X_train, y_train, result)

        rmse = round(mean_squared_error(_ytrue, _yhat, squared=False), 3)
        mape = round(mean_absolute_percentage_error(_ytrue, _yhat), 3)
        r2 = round(r2_score(_ytrue, _yhat), 3)
        logger.info(f"[val]  RMSE={rmse}, MAPE={mape:.3f}, R2={r2:.3f}")

        rmse = round(mean_squared_error(self.pr.ytrue, self.pr.yhat, squared=False), 3)
        mape = round(mean_absolute_percentage_error(self.pr.ytrue, self.pr.yhat), 3)
        r2 = round(r2_score(self.pr.ytrue, self.pr.yhat), 3)
        logger.info(f"[test] RMSE={rmse}, MAPE={mape:.3f}, R2={r2:.3f}")

        return self.pr

    def crossvalidate(self, X, y, result):
        # Get y_true and y_hat for each cross validation iteration. Because of temporal
        # dependency, each split is always equal
        idxs = np.array([], dtype=int)
        yhat = np.array([], dtype=float)
        ytrue = np.array([], dtype=float)
        timestamps = np.array([], dtype=np.datetime64)

        # 1092, 17
        # 1092,
        for i, (trainidxs, testidxs) in enumerate(self.tscv.split(X, y)):
            X_train, X_test = X[trainidxs], X[testidxs]
            y_train, y_test = y[trainidxs], y[testidxs]

            # Fit a model
            model = self.method(**self.defaults, **result.best_params_)
            model.fit(X_train, y_train)

            # Predict values
            # FIXME: ValueError: Found array with 0 sample(s) (shape=(0, 17)) while a
            #  minimum of 1 is required.
            try:
                y_hat = model.predict(X_test)
            except ValueError as err:
                print()

            # TODO: assert fold results are the same

            idxs = np.append(idxs, np.full(shape=y_hat.size, fill_value=i))
            timestamps = np.append(timestamps, self.df.index[testidxs].to_numpy())

            ytrue_ = src.utils.rescaletarget(self._scaler, y_test)
            ytrue = np.append(ytrue, ytrue_)

            yhat_ = src.utils.rescaletarget(self._scaler, y_hat)
            yhat = np.append(yhat, yhat_)

        return idxs, ytrue, yhat, timestamps

    def predict(self, X):
        return self.model.predict(X)


class ProcessorResults:
    def __init__(self, name, model, vm, trainw, n_splits, path):
        self.id = f"{name}-{vm}-{trainw}-{n_splits}"
        self.model = model
        self.output = path

        self.vm = vm  # Validation method
        self.trainw = trainw  # Train window
        self.n_splits = n_splits  # Number of splits for cross validation

        self.yhat = np.array([], dtype=float)
        self.ytrue = np.array([], dtype=float)
        self.split = np.array([], dtype=int)
        self.timestamp = np.array([], dtype=np.datetime64)

    def add(self, split, yhat, ytrue, timestamp):
        self.split = np.append(self.split, np.full(yhat.size, fill_value=split))
        self.yhat = np.append(self.yhat, yhat)
        self.ytrue = np.append(self.ytrue, ytrue)
        self.timestamp = np.append(self.timestamp, timestamp)

    def save(self):
        if not os.path.isdir(self.output):
            os.makedirs(self.output, exist_ok=True)

        # Create dataframe
        data = {"ytrue": self.ytrue, "yhat": self.yhat, "split": self.split}
        df = pd.DataFrame(data=data, index=self.timestamp)

        # Save predictions
        dst = os.path.join(self.output, f"{self.id}.csv")
        df.to_csv(dst, index=True, index_label="timestamp")

        # Save model
        dst = os.path.join(self.output, f"{self.id}.pickle")
        with open(dst, "wb") as f:
            pickle.dump(self, f)

    def getdst(self):
        return os.path.join(self.output, f"{self.id}.pickle")


class LRProcessor(Processor):
    def __init__(self, id, df, vm, trainw, n_splits, output):
        super().__init__(id, df, vm, trainw, n_splits, output)
        self.method = LinearRegression
        self.defaults = dict()
        self.space = {"fit_intercept": [True, False]}
        self.model = LinearRegression(**self.defaults)


class KNNProcessor(Processor):
    def __init__(self, id, df, vm, trainw, n_splits, output):
        super().__init__(id, df, vm, trainw, n_splits, output)
        self.method = KNeighborsRegressor
        self.defaults = dict(metric="minkowski", p=2, weights="distance")
        self.space = dict(n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
        self.model = KNeighborsRegressor(**self.defaults)


class SVRProcessor(Processor):
    def __init__(self, id, df, vm, trainw, n_splits, output):
        super().__init__(id, df, vm, trainw, n_splits, output)
        self.method = SVR
        self.defaults = dict()

        # NOTE: complete
        # self.space = dict(
        #     C=[0.05, 0.1, 0.5, 1],
        #     epsilon=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        #     gamma=[0.0001, 0.001],
        #     kernel=["linear", "poly", "rbf", "sigmoid"],
        # )

        # NOTE: partial
        self.space = dict(
            C=[0.1, 0.5, 1.0],
            epsilon=[0.0001, 0.01, 0.1, 1.0],
            gamma=[0.0001, 0.001],
            kernel=["linear", "poly", "rbf", "sigmoid"],
        )

        self.model = SVR(**self.defaults)


class MLPProcessor(Processor):
    def __init__(self, id, df, vm, trainw, n_splits, output):
        super().__init__(id, df, vm, trainw, n_splits, output)
        self.method = MLPRegressor
        self.defaults = dict(shuffle=False, verbose=False, random_state=1)

        # NOTE: complete
        # self.space = dict(
        #     hidden_layer_sizes=[
        #         # (8,), (16,), (32,),
        #         # (8, 2), (16, 2), (32, 2),
        #         # (8, 4), (16, 4), (32, 4),
        #         # (8, 8), (16, 8), (32, 8),
        #
        #         (13,), (13, 2),
        #     ],
        #     activation=["logistic", "tanh", "relu"],
        #     solver=["lbfgs", "sgd", "adam"],
        #     # alpha=[0.0001, 0.05, 0.01, 0.1],
        #     alpha=[0.0001, 0.001],
        #     learning_rate=["constant", "invscaling", "adaptive"],
        #     # learning_rate_init=[0.001, 0.05, 0.01, 0.1],
        #     learning_rate_init=[0.0001, 0.001],
        #     max_iter=[1000],
        #     tol=[1e-4],
        #     # momentum=[0.9, 0.99],
        #     momentum=[0.9],
        #     early_stopping=[True],
        # )

        # NOTE: partial
        self.space = dict(
            hidden_layer_sizes=[(13,), (21,), (29,)],
            activation=["logistic", "tanh", "relu"],
            solver=["adam"],
            learning_rate_init=[0.001, 0.01],
            max_iter=[1000],
            tol=[1e-4],
            momentum=[0.9],
            early_stopping=[True],
        )

        self.model = MLPRegressor(**self.defaults)
