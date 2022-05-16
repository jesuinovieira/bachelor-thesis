import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import src.utils
import src.tscv as tscv

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, name, df, cv, trainw, testw, output):
        self._name = name
        self.method = None
        self.model = None
        self.space = None
        self.defaults = None

        self.df = df
        self._scaler = MinMaxScaler()

        self.cv = tscv.ExpandingWindow if cv.upper() == "EW" else tscv.SlidingWindow
        self.trainw = trainw
        self.testw = testw
        self.pr = ProcessorResults(
            self._name, self.model, cv, self.trainw, self.testw, path=output
        )

    def transform(self):
        self._scaler.fit(self.df)
        self.df[self.df.columns] = self._scaler.transform(self.df)

    def fit(self):
        # TODO: fit() -> model selection
        # TODO: evaluate() -> model evaluation
        # TODO: save and plot val metrics too

        # Split data into train and test set (validation set is included in train)
        # TODO: train_test_split by timestamp
        train, test = train_test_split(self.df, test_size=0.2475, shuffle=False)

        train = train.to_numpy()
        X_train, y_train = train[:, 1:], train[:, 0].T

        # Model selection
        n_jobs = -1
        scoring = "neg_root_mean_squared_error"
        # TODO: multiple scores
        # scoring = [
        #     "neg_root_mean_squared_error",
        #     "neg_mean_absolute_percentage_error",
        #     "r2_score"
        # ]

        rows, _ = X_train.shape
        ms_cv = self.cv(n_samples=rows, trainw=self.trainw, testw=self.testw)
        search = GridSearchCV(
            self.model, self.space, cv=ms_cv, scoring=scoring, n_jobs=n_jobs, verbose=10
        )
        result = search.fit(X_train, y_train)

        # Fit selected model in train data
        # model = self.method(**self.defaults, **result.best_params_)
        # model.fit(X_train, y_train)
        # Evaluate on test data
        # y_hat = model.predict(X_test)

        logger.info(f"Best params for '{self._name}': {result.best_params_}")

        timestamps = self.df.index.to_numpy()

        rows, _ = X_train.shape
        ms_cv = self.cv(n_samples=rows, trainw=self.trainw, testw=self.testw)
        model = self.method(**self.defaults, **result.best_params_)
        ms_iteration, ms_timestamp, ms_ytrue, ms_yhat = tscv.crossvalidate(
            model, ms_cv, X_train, y_train, timestamps
        )

        ms_ytrue = src.utils.rescaletarget(self._scaler, ms_ytrue)
        ms_yhat = src.utils.rescaletarget(self._scaler, ms_yhat)

        # self.pr.add(
        #     type="validation",
        #     split=_fold,
        #     yhat=_yhat,
        #     ytrue=_ytrue,
        #     timestamp=_timestamp
        # )

        rmse = round(mean_squared_error(ms_ytrue, ms_yhat, squared=False), 2)
        mape = round(mean_absolute_percentage_error(ms_ytrue, ms_yhat), 2)
        r2 = round(r2_score(ms_ytrue, ms_yhat), 2)
        logger.info(f"[val]  RMSE={rmse}, MAPE={mape:.2f}, R2={r2:.2f}")

        # ==============================================================================

        # Model evaluation
        data = self.df.to_numpy()
        X, y = data[:, 1:], data[:, 0].T
        timestamps = self.df.index.to_numpy()

        rows, _ = X.shape
        me_cv = self.cv(n_samples=rows, trainw=1092, testw=self.testw)
        model = self.method(**self.defaults, **result.best_params_)
        me_iteration, me_timestamp, me_ytrue, me_yhat = tscv.crossvalidate(
            model, me_cv, X, y, timestamps
        )

        me_ytrue = src.utils.rescaletarget(self._scaler, me_ytrue)
        me_yhat = src.utils.rescaletarget(self._scaler, me_yhat)

        self.pr.add(
            type="test",
            split=me_iteration,
            timestamp=me_timestamp,
            yhat=me_yhat,
            ytrue=me_ytrue,
        )

        rmse = round(mean_squared_error(me_ytrue, me_yhat, squared=False), 2)
        mape = round(mean_absolute_percentage_error(me_ytrue, me_yhat), 2)
        r2 = round(r2_score(me_ytrue, me_yhat), 2)
        logger.info(f"[test] RMSE={rmse}, MAPE={mape:.2f}, R2={r2:.2f}")

        self.pr.save()
        return self.pr

    def predict(self, X):
        return self.model.predict(X)


# NOTE: Ordinary Least Squares
class LRProcessor(Processor):
    def __init__(self, id, df, cv, trainw, testw, output):
        super().__init__(id, df, cv, trainw, testw, output)
        self.method = LinearRegression
        self.defaults = dict(fit_intercept=True)
        self.space = dict()
        self.model = LinearRegression(**self.defaults)


class KNNProcessor(Processor):
    def __init__(self, id, df, cv, trainw, testw, output):
        super().__init__(id, df, cv, trainw, testw, output)
        self.method = KNeighborsRegressor
        self.defaults = dict(algorithm="auto")

        # NOTE: A escolha da proximidade utilizada é fundamental para o kNN!
        # - Algorithm (brute, ball_tree or kd_tree) don't change the result, right?
        # - Any other metric than euclidean distance to grid search? (It's fast!)

        self.space = dict(
            n_neighbors=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
            weights=["uniform", "distance"],
            metric=[
                "euclidean",
                "l2",
                "l1",
                "manhattan",
                "cityblock",
                "braycurtis",
                "canberra",
                "chebyshev",
                "correlation",
                "cosine",
                "hamming",
                "minkowski",
                "nan_euclidean",
                # "haversine" (only valid for 2d)
                # "wminkowski" (requires a weight vector `w` to be given)
                # "yule" (data was converted to boolean)
                # "sokalsneath" (data was converted to boolean)
                # "sokalmichener" (data was converted to boolean)
                # "sqeuclidean" ('V' parameter is required when Y is passed)
                # "russellrao" (data was converted to boolean)
                # "rogerstanimoto" (data was converted to boolean)
                # "seuclidean" ('V' parameter is required when Y is passed)
                # "matching" (data was converted to boolean)
                # "mahalanobis" ('VI' parameter is required when Y is passed)
                # "kulsinski" (data was converted to boolean)
                # "jaccard" (data was converted to boolean)
                # "dice" (data was converted to boolean)
            ],
        )
        self.model = KNeighborsRegressor(**self.defaults)


class SVRProcessor(Processor):
    def __init__(self, id, df, cv, trainw, testw, output):
        super().__init__(id, df, cv, trainw, testw, output)
        self.method = SVR
        self.defaults = dict(cache_size=500)

        # TODO: improve epsilon, gamma and tol (?)
        # https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
        # - C: if you have a lot of noisy observations you should decrease it.
        #   Decreasing C corresponds to more regularization
        # - One is advised to use GridSearchCV with C and gamma spaced exponentially far
        #   apart to choose good values
        # - In practice, a logarithmic grid from 10^-3 to 10^+3 is usually sufficient
        # - 'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]

        # self.space = dict(
        #     kernel=["linear", "poly", "rbf", "sigmoid"],
        #     C=[0.1, 0.5, 1.0, 1.5, 2.0],
        #     epsilon=[0.01, 0.05, 0.1],
        #     gamma=["scale", "auto"],
        #     tol=[1e-3],
        # )

        self.space = dict(
            kernel=["linear", "poly", "rbf", "sigmoid"],
            C=[0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            epsilon=[0.01, 0.05, 0.1, 0.5, 1.0],
            gamma=["scale", "auto"],
            tol=[1e-3]
        )

        self.model = SVR(**self.defaults)


class MLPProcessor(Processor):
    def __init__(self, id, df, cv, trainw, testw, output):
        super().__init__(id, df, cv, trainw, testw, output)
        self.method = MLPRegressor
        self.defaults = dict(shuffle=False, random_state=16)

        # TODO: MLP with Keras?
        # TODO: training and validation error plots

        # https://scikit-learn.org/stable/modules/neural_networks_supervised.html#mlp-tips
        # https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html#sphx-glr-auto-examples-neural-networks-plot-mlp-alpha-py
        #
        # Hence do not loop through with different max_iterations, try to tweak the tol
        # and n_iter_no_change if you want to avoid the overfitting.

        # self.space = dict(
        #     hidden_layer_sizes=[(13,)],
        #     activation=["tanh", "relu"],
        #     solver=["lbfgs"],
        #     alpha=[0.05, 0.1],
        #     momentum=[0.9],
        #
        #     learning_rate=["constant"],
        #     learning_rate_init=[0.005, 0.001],
        #
        #     max_iter=[500],
        #     n_iter_no_change=[10],
        #     tol=[1e-4],
        #     early_stopping=[True],
        # )

        # hidden_layer_sizes=[
        #     # (8,), (16,), (32,),
        #     # (8, 2), (16, 2), (32, 2),
        #     # (8, 4), (16, 4), (32, 4),
        #     # (8, 8), (16, 8), (32, 8),
        #
        #     (13,), (13, 2),
        # ],

        self.space = dict(
            hidden_layer_sizes=[(13,), (21,), (29,)],
            activation=["logistic", "tanh", "relu"],
            solver=["lbfgs", "sgd", "adam"],
            alpha=[0.01, 0.5, 0.1],
            momentum=[0.9, 0.95],

            learning_rate=["constant", "invscaling", "adaptive"],
            learning_rate_init=[0.001, 0.005, 0.01],

            max_iter=[500],
            n_iter_no_change=[10],
            tol=[1e-4],
            early_stopping=[True],
        )

        self.model = MLPRegressor(**self.defaults)


class ProcessorResults:
    def __init__(self, name, model, cv, trainw, testw, path):
        self.id = f"{name}-{cv}-{trainw}-{testw}"
        self.model = model
        self.output = path

        self.cv = cv  # Cross validation method
        self.trainw = trainw  # Train window
        self.n_splits = testw  # Test window

        self.yhat = np.array([], dtype=float)
        self.ytrue = np.array([], dtype=float)
        self.split = np.array([], dtype=int)
        self.type = np.array([], dtype=str)
        self.timestamp = np.array([], dtype=np.datetime64)

    def add(self, type, split, yhat, ytrue, timestamp):
        self.type = np.append(self.type, np.full(yhat.size, fill_value=type))
        self.split = np.append(self.split, np.full(yhat.size, fill_value=split))
        self.yhat = np.append(self.yhat, yhat)
        self.ytrue = np.append(self.ytrue, ytrue)
        self.timestamp = np.append(self.timestamp, timestamp)

    def save(self):
        if not os.path.isdir(self.output):
            os.makedirs(self.output, exist_ok=True)

        # Create dataframe
        data = dict(ytrue=self.ytrue, yhat=self.yhat, split=self.split, type=self.type)
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
