import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import src.utils

# TODO: orchestrate model training (googleit)


class Processor:
    def __init__(self, name, df, dbname, vm, trainw, n_splits):
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
        # scoring = "neg_mean_absolute_percentage_error"
        scoring = "neg_root_mean_squared_error"
        n_jobs = -1

        search = GridSearchCV(
            self.model, self.space, cv=self.tscv, scoring=scoring, n_jobs=n_jobs,
            # verbose=True
        )
        result = search.fit(X, y)

        # Get y_true and y_hat for each cross validation iteration
        # Only works because each split is always equal, different from KFoldValidation
        timestamps = np.array([], dtype=np.datetime64)
        for i, (trainidxs, testidxs) in enumerate(self.tscv.split(X, y)):
            X_train, X_test = X[trainidxs], X[testidxs]
            y_train, y_test = y[trainidxs], y[testidxs]

            # Fit a model
            model = self.method(**self.defaults, **result.best_params_)
            model.fit(X_train, y_train)

            # Predict values
            y_hat = model.predict(X_test)

            # Temporary: assert results are equal to grid search
            # --------------------------------------------------------------------------
            # from sklearn.metrics import mean_absolute_percentage_error
            # mape = mean_absolute_percentage_error(y_test, y_hat)
            # from sklearn.metrics import mean_squared_error
            # mape = mean_squared_error(y_test, y_hat, squared=True)
            # sklearnmape = abs(
            #     result.cv_results_[f"split{i}_test_score"][result.best_index_]
            # )
            # try:
            #     assert round(mape, 10) == round(sklearnmape, 10)
            # except AssertionError as err:
            #     print(err)
            #     print(mape, sklearnmape)
            # --------------------------------------------------------------------------

            # Save results
            self.pr.add(
                split=i,
                yhat=src.utils.rescaletarget(self._scaler, y_hat),
                ytrue=src.utils.rescaletarget(self._scaler, y_test),
            )

            new = self.df.index[testidxs].to_numpy()
            timestamps = np.append(timestamps, new)

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
        self.split = np.array([], dtype=int)
        self.timestamp = np.array([], dtype=float)

    def add(self, split, yhat, ytrue):
        assert yhat.size == ytrue.size
        self.yhat = np.append(self.yhat, yhat)
        self.ytrue = np.append(self.ytrue, ytrue)
        self.split = np.append(self.split, np.full(yhat.size, fill_value=split))

    def save(self, timestamps):
        if not os.path.isdir(self.output):
            os.makedirs(self.output, exist_ok=True)

        if not self.timestamp.size:
            self.timestamp = timestamps

        assert (
            self.yhat.size == self.ytrue.size == self.timestamp.size == self.split.size
        )

        # Save predictions
        data = {"ytrue": self.ytrue, "yhat": self.yhat, "split": self.split}
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
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.method = LinearRegression
        self.defaults = dict()
        self.space = {"fit_intercept": [True, False]}
        self.model = LinearRegression(**self.defaults)


class KNNProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.method = KNeighborsRegressor
        self.defaults = dict(metric="minkowski", p=2, weights="distance")
        self.space = dict(n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19])
        self.model = KNeighborsRegressor(**self.defaults)


class SVRProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.method = SVR
        self.defaults = dict()
        self.space = dict(
            C=[0.05, 0.1, 0.5, 1],
            epsilon=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            # gamma=[0.0001, 0.001],
            kernel=["linear", "poly", "rbf", "sigmoid"]
        )
        self.model = SVR(**self.defaults)
        # self.model = SVR(kernel="rbf", C=1, epsilon=0.1)


class MLPProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.method = MLPRegressor
        self.defaults = dict(shuffle=False, verbose=False, random_state=1)

        # Examples
        # ------------------------------------------------------------------------------
        # parameter_space = {
        #     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        #     'activation': ['tanh', 'relu'],
        #     'solver': ['sgd', 'adam'],
        #     'alpha': [0.0001, 0.05],
        #     'learning_rate': ['constant','adaptive'],
        # }
        #
        # parameter_space = {
        #     'hidden_layer_sizes': [(368,), (555,), (100,)],
        #     'activation': ['identity', 'logistic', 'relu'],
        #     'solver': ['sgd', 'adam'],
        #     'alpha': [0.0001, 0.05],
        #     'learning_rate': ['constant','adaptive'],
        #     'max_iter': ['200', '1000', '5000', '10000']
        # }

        # self.space = dict(
        #     hidden_layer_sizes=[
        #         (8,), (16,), (32,),
        #         (8, 2), (16, 2), (32, 2),
        #         (8, 4), (16, 4), (32, 4),
        #         (8, 8), (16, 8), (32, 8),
        #     ],
        #     activation=["identity", "logistic", "tanh", "relu"],
        #     solver=["lbfgs", "sgd", "adam"],
        #     alpha=[0.0001, 0.05],
        #     learning_rate=["constant", "invscaling", "adaptive"],
        #     learning_rate_init=[0.001, 0.01],
        #     max_iter=[50, 100, 1000],
        #     tol=[1e-4],
        #     momentum=[0.9, 0.99],
        #     early_stopping=[True, False],
        # )
        self.space = dict(
            hidden_layer_sizes=[
                (11,), (13,), (15,), (17,), (19,), (21,), (23,), (25,), (27,), (29,)
            ],
            activation=["tanh", "relu"],
            solver=["adam"],
            learning_rate_init=[0.001, 0.01],
            max_iter=[1000],
            tol=[1e-4],
            momentum=[0.9, 0.99],
            early_stopping=[True, False]
        )
        self.model = MLPRegressor(**self.defaults)

        # self.model = MLPRegressor(
        #     hidden_layer_sizes=(17, ),
        #     activation="relu",
        #     learning_rate_init=0.01,
        #     max_iter=1000,
        #     tol=1e-4,
        #     momentum=0.9,
        #     early_stopping=True,
        #     random_state=1,
        # )
