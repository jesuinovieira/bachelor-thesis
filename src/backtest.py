import math

import numpy as np

# TODO: better handle last iteration. For example, if there are 10 iterations and 101
#  samples, the last iteration must have 11 observations.
# TODO: make code reusable (SW and EW are very similar)
# TODO: save vector of metrics computed for each iteration


class SlidingWindow:
    def __init__(self, n_samples, trainw, testw):
        self.n_samples = n_samples
        self.trainw = trainw
        self.testw = testw

        self.n_splits = math.ceil((self.n_samples - self.trainw) / testw)

        assert n_samples != self.trainw
        assert self.testw > 0

    def split(self, X, y=None, groups=None):
        for i, k in enumerate(range(self.trainw, self.n_samples, self.testw)):
            trainidxs = slice(k - self.trainw, k)
            testidxs = slice(k, k + self.testw)

            if i + 1 == self.n_splits:
                testidxs = slice(k, self.n_samples)

            yield trainidxs, testidxs

            if i + 1 == self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class ExpandingWindow:
    def __init__(self, n_samples, trainw, testw):
        self.n_samples = n_samples
        self.trainw = trainw
        self.testw = testw

        self.n_splits = math.ceil((self.n_samples - self.trainw) / testw)

        assert n_samples != self.trainw
        assert self.testw > 0

    def split(self, X, y=None, groups=None):
        for i, k in enumerate(range(self.trainw, self.n_samples, self.testw)):
            trainidxs = slice(0, k)
            testidxs = slice(k, k + self.testw)

            if i + 1 == self.n_splits:
                testidxs = slice(k, self.n_samples)

            yield trainidxs, testidxs

            if i + 1 == self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def backtest(model, tscv, X, y, timestamps):
    # TODO: move to utils
    # Get y_true and y_hat for each cross validation iteration. Because of temporal
    # dependency, each split is always equal

    retval = dict(
        iteration=np.array([], dtype=int),
        timestamp=np.array([], dtype=np.datetime64),
        yhat=np.array([], dtype=float),
        ytrue=np.array([], dtype=float),
    )

    for i, (trainidxs, testidxs) in enumerate(tscv.split(X, y)):
        X_train, X_test = X[trainidxs], X[testidxs]
        y_train, y_test = y[trainidxs], y[testidxs]

        # Fit a model
        # model = self.method(**self.defaults, **result.best_params_)
        model.fit(X_train, y_train)

        # Predict values
        y_hat = model.predict(X_test)

        # TODO: assert fold results are the same

        iteration = np.full(shape=y_hat.size, fill_value=i)
        retval["iteration"] = np.append(retval["iteration"], iteration)
        retval["timestamp"] = np.append(retval["timestamp"], timestamps[testidxs])
        retval["ytrue"] = np.append(retval["ytrue"], y_test)
        retval["yhat"] = np.append(retval["yhat"], y_hat)

    # NOTE: need to be rescaled
    return retval["iteration"], retval["timestamp"], retval["ytrue"], retval["yhat"]
