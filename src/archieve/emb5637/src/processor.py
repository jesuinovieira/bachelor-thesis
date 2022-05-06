import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from src.utils import cvsplit
from src.utils import rescaletarget


# TODO: orchestrate model training (googleit)


class Processor:
    def __init__(self, name, df, dbname, vm, trainw, testw):
        self._name = name

        self.df = df
        self.model = None
        self._scaler = MinMaxScaler()

        self.vm = vm
        self.trainw = trainw
        self.testw = testw
        self.pr = ProcessorResults(
            self._name, self.model, self.vm, self.trainw, self.testw, dbname
        )

    def transform(self):
        self._scaler.fit(self.df)
        self.df[self.df.columns] = self._scaler.transform(self.df)

    def fit(self):
        # Cross-validation
        for X_train, X_test, y_train, y_test in cvsplit(
            self.trainw, self.testw, self.vm, self.df
        ):
            self.model.fit(X_train, y_train)
            y_hat = self.predict(X_test)

            # Rescale the target and save results
            self.pr.add(
                yhat=rescaletarget(self._scaler, y_hat),
                ytrue=rescaletarget(self._scaler, y_test),
            )

        timestamps = self.df.index[self.trainw :].to_numpy()
        self.pr.save(timestamps)

        return self.pr

    def predict(self, X):
        return self.model.predict(X)


class ProcessorResults:
    def __init__(self, name, model, vm, trainw, testw, dbname, path="output/processor"):
        self.id = f"{dbname}-{name}-{vm}-{trainw}-{testw}"
        self.model = model
        self.output = path

        self.vm = vm  # Validation method
        self.trainw = trainw  # Train window
        self.testw = testw  # Test window
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
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.model = LinearRegression()


class KNNProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.model = KNeighborsRegressor(
            n_neighbors=11, weights="distance", metric="minkowski", p=2
        )


class SVRProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.model = SVR(kernel="rbf", C=1, epsilon=0.1)


class MLPProcessor(Processor):
    def __init__(self, id, df, dbname, vm, trainw, testw):
        super().__init__(id, df, dbname, vm, trainw, testw)
        self.model = MLPRegressor(
            hidden_layer_sizes=(17,),
            activation="relu",
            learning_rate_init=0.01,
            max_iter=1000,
            tol=1e-4,
            momentum=0.9,
            early_stopping=True,
            random_state=1,
        )
