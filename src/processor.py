import logging
import math
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import src.utils
import src.backtest

logger = logging.getLogger(__name__)


def _getbacktest(df, wtype, wsize, forwhat):
    testw = 7

    if wtype == "EW" and wsize == "7D":
        if forwhat == "model-selection":
            trainw = len(df.loc["2016-01-01":"2017-12-31"])
        else:
            trainw = len(df.loc["2016-01-01":"2018-12-31"])

        n_samples = len(df)
        return df, src.backtest.ExpandingWindow(n_samples, trainw, testw)

    if wtype == "SW" and wsize == "1Y7D":
        if forwhat == "model-selection":
            df = df.loc["2017-01-01":]
            trainw = len(df.loc["2017-01-01":"2017-12-31"])
        else:
            df = df.loc["2018-01-01":]
            trainw = len(df.loc["2018-01-01":"2018-12-31"])

        n_samples = len(df)
        return df, src.backtest.SlidingWindow(n_samples, trainw, testw)

    if wtype == "SW" and wsize == "2Y7D":
        if forwhat == "model-selection":
            trainw = len(df.loc["2016-01-01":"2017-12-31"])
        else:
            df = df.loc["2017-01-01":]
            trainw = len(df.loc["2017-01-01":"2018-12-31"])

        n_samples = len(df)
        return df, src.backtest.SlidingWindow(n_samples, trainw, testw)


class Processor:
    def __init__(self, name, df, backtest, output):
        self._name = name
        self.method = None
        self.model = None
        self.space = None
        self.defaults = None
        self.best_params = None

        self.df = df
        self._scaler = MinMaxScaler()

        # FIXME: backtest is hardcoded, only the following configurations are supported
        assert backtest in ["EW-7D", "SW-1Y7D", "SW-2Y7D"]

        wtype, wsize = backtest.split("-")
        self.wtype = wtype
        self.wsize = wsize

        self.pr = ProcessorResults(name, df, backtest, path=output)

    def transform(self):
        self._scaler.fit(self.df)
        self.df[self.df.columns] = self._scaler.transform(self.df)

    def fit(self):
        # TODO: train_test_split by timestamp
        # TODO: multi score grid search
        self._ms()
        self._me()

        return self.pr

    def _ms(self):
        # Split data and construct backtest object
        train, test = train_test_split(self.df, test_size=0.25, shuffle=False)

        train, cv = _getbacktest(train, self.wtype, self.wsize, "model-selection")
        timestamps = train.index.to_numpy()

        data = train.to_numpy()
        X_train, y_train = data[:, 1:], data[:, 0].T

        # Grid search
        scoring = "neg_root_mean_squared_error"
        # scoring = "neg_mean_absolute_error"
        search = GridSearchCV(
            self.model, self.space, cv=cv, scoring=scoring, n_jobs=-1, verbose=10
        )
        result = search.fit(X_train, y_train)
        self.best_params = result.best_params_

        # Backtest selected model on validation set
        model = self.method(**self.defaults, **self.best_params)
        iteration, ts, ytrue, yhat = src.backtest.backtest(
            model, cv, X_train, y_train, timestamps
        )

        # Rescale, save and output results
        ytrue = src.utils.rescaletarget(self._scaler, ytrue)
        yhat = src.utils.rescaletarget(self._scaler, yhat)

        self.pr.add(
            split="val", iteration=iteration, yhat=yhat, ytrue=ytrue, timestamp=ts
        )

        rmse = round(mean_squared_error(ytrue, yhat, squared=False), 2)
        mape = round(mean_absolute_percentage_error(ytrue, yhat), 2)
        mae = round(mean_absolute_error(ytrue, yhat), 2)
        r2 = round(r2_score(ytrue, yhat), 2)

        logger.info(f"Best params for '{self._name}': {result.best_params_}")
        logger.info(f"[val]  RMSE={rmse}, MAPE={mape:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

    def _me(self):
        # Split data and construct backtest object
        df, cv = _getbacktest(self.df, self.wtype, self.wsize, "model-evaluation")
        timestamps = df.index.to_numpy()

        data = df.to_numpy()
        X, y = data[:, 1:], data[:, 0].T

        # Backtest selected model on test set
        model = self.method(**self.defaults, **self.best_params)
        iteration, ts, ytrue, yhat = src.backtest.backtest(model, cv, X, y, timestamps)

        # Rescale, save and output results
        ytrue = src.utils.rescaletarget(self._scaler, ytrue)
        yhat = src.utils.rescaletarget(self._scaler, yhat)

        self.pr.add(
            split="test", iteration=iteration, timestamp=ts, yhat=yhat, ytrue=ytrue
        )

        rmse = round(mean_squared_error(ytrue, yhat, squared=False), 2)
        mape = round(mean_absolute_percentage_error(ytrue, yhat), 2)
        mae = round(mean_absolute_error(ytrue, yhat), 2)
        r2 = round(r2_score(ytrue, yhat), 2)

        logger.info(f"[test] RMSE={rmse}, MAPE={mape:.2f}, MAE={mae:.2f}, R2={r2:.2f}")
        self.pr.save(model)

    def predict(self, X):
        return self.model.predict(X)


# NOTE: Ordinary Least Squares
class LRProcessor(Processor):
    def __init__(self, id, df, backtest, output):
        super().__init__(id, df, backtest, output)
        self.method = LinearRegression
        self.defaults = dict(fit_intercept=True)
        self.space = dict()
        self.model = LinearRegression(**self.defaults)


class KNNProcessor(Processor):
    def __init__(self, id, df, backtest, output):
        super().__init__(id, df, backtest, output)
        self.method = KNeighborsRegressor
        self.defaults = dict(algorithm="auto")

        self.space = dict(
            n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
            weights=["uniform", "distance"],
            metric=["euclidean", "manhattan"],
        )

        # self.space = dict(
        #     n_neighbors=[7, 9, 11, 13],
        #     weights=["uniform", "distance"],
        #     metric=["euclidean", "manhattan"],
        # )

        self.model = KNeighborsRegressor(**self.defaults)


class SVRProcessor(Processor):
    def __init__(self, id, df, backtest, output):
        super().__init__(id, df, backtest, output)
        self.method = SVR
        self.defaults = dict(cache_size=500)

        # C: if you have a lot of noisy observations you should decrease it.
        #
        # One is advised to use GridSearchCV with C and gamma spaced exponentially far
        # apart to choose good values
        #
        # In practice, a logarithmic grid from 10^-3 to 10^+3 is usually sufficient. If
        # the best parameters lie on the boundaries of the grid, it can be extended in
        # that direction in a subsequent search.

        self.space = dict(
            kernel=["linear", "poly", "rbf", "sigmoid"],
            C=np.logspace(math.log10(0.001), math.log10(100), 10),
            epsilon=np.logspace(math.log10(0.001), math.log10(0.1), 10),
            gamma=["scale", "auto"],
        )

        # self.space = dict(
        #     kernel=["rbf"],
        #     C=[0.5, 1.0, 1.5],
        #     epsilon=[0.05],
        #     gamma=["scale"],
        # )

        self.model = SVR(**self.defaults)


class MLPProcessor(Processor):
    def __init__(self, id, df, backtest, output):
        super().__init__(id, df, backtest, output)
        self.method = MLPRegressor
        # fmt: off
        self.defaults = dict(
            shuffle=False, early_stopping=True, max_iter=1000, random_state=16,
            batch_size=16
        )
        # fmt: on

        # TODO: MLP with Keras?
        # TODO: training and validation error plots
        #
        # Googlit: tune hyperparameters for regression machine learning algorithms
        # https://github.com/hyperopt/hyperopt-sklearn
        #
        # https://scikit-learn.org/stable/modules/neural_networks_supervised.html#mlp-tips
        # https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
        #
        # Hence do not loop through with different max_iterations, try to tweak the tol
        # and n_iter_no_change if you want to avoid the overfitting.

        self.space = dict(
            hidden_layer_sizes=[(11,), (13,), (15,), (17,), (19,), (21,), (23,)],
            activation=["tanh", "relu"],
            solver=["lbfgs", "sgd", "adam"],
            alpha=np.logspace(math.log10(0.01), math.log10(1.0), 10),
            learning_rate_init=np.logspace(math.log10(0.0001), math.log10(0.01), 5),
        )

        # self.space = dict(
        #     hidden_layer_sizes=[(13,), (15,), (21,)],
        #     activation=["tanh", "relu"],
        #     solver=["lbfgs"],
        #     alpha=np.logspace(math.log10(0.01), math.log10(1.0), 5),
        # )

        self.model = MLPRegressor(**self.defaults)


class ProcessorResults:
    def __init__(self, name, df, backtest, path):
        self.id = f"{name}-{backtest}"
        self.df = df
        self.backtest = backtest
        self.output = path
        self.model = None

        self.iteration = np.array([], dtype=int)
        self.timestamp = np.array([], dtype=np.datetime64)
        self.yhat = np.array([], dtype=float)
        self.ytrue = np.array([], dtype=float)
        self.split = np.array([], dtype=str)

    def add(self, split, iteration, yhat, ytrue, timestamp):
        split = np.full(yhat.size, fill_value=split)

        self.iteration = np.append(self.iteration, iteration)
        self.timestamp = np.append(self.timestamp, timestamp)
        self.yhat = np.append(self.yhat, yhat)
        self.ytrue = np.append(self.ytrue, ytrue)
        self.split = np.append(self.split, split)

    def save(self, model):
        if not os.path.isdir(self.output):
            os.makedirs(self.output, exist_ok=True)

        self.model = model
        data = dict(
            ytrue=self.ytrue, yhat=self.yhat, iteration=self.iteration, split=self.split
        )

        df = pd.DataFrame(data=data, index=self.timestamp)
        self.df = pd.concat([self.df, df], axis=1)
        self.df.split = self.df.split.fillna("train")

        # Save predictions
        dst = os.path.join(self.output, f"{self.id}.csv")
        df.to_csv(dst, index=True, index_label="timestamp")

        # Save model
        dst = os.path.join(self.output, f"{self.id}.pickle")
        with open(dst, "wb") as f:
            pickle.dump(self, f)

    def getdst(self):
        return os.path.join(self.output, f"{self.id}.pickle")
