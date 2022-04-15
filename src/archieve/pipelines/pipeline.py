import os

import numpy as np
import pandas as pd

import models
import utils


# TODO: PipelineResults
class ModelResults:
    def __init__(self, name, model, vm, trainw, testw, db):
        self.name = name
        self.model = model

        self.vm = vm  # Validation method
        self.trainw = trainw  # Train window
        self.testw = testw  # Test window
        self.db = db

        self.yhat = np.array([], dtype=float)
        self.ytrue = np.array([], dtype=float)

    def add(self, yhat, ytrue):
        self.yhat = np.append(self.yhat, yhat)
        self.ytrue = np.append(self.ytrue, ytrue)

    def save(self, basepath="pipelines/results"):
        if not os.path.isdir(basepath):
            os.makedirs(basepath, exist_ok=True)

        file = (
            f"{basepath}/"
            f"{self.db}-{self.name}-{self.vm}-{self.trainw}-{self.testw}"
            f".csv"
        )
        df = pd.DataFrame({"ytrue": self.ytrue, "yhat": self.yhat})
        df.to_csv(file, index=False)


def main():
    for db in utils.fs2attributes.keys():
        print(f"Running models for {db}")

        # TODO: a folder for each training window! Must be evaluated in a separated way?
        for vm, trainw, testw in vmconfigs:
            print(f"\tConfig: {vm}-{trainw}-{testw}")
            mresults = []

            # TODO: custom transformation for each method? Then each one will have your
            #  loop..

            # Initialize linear regression models
            for i, model in enumerate(models.lr.get()):
                item = ModelResults(f"LR{i + 1}", model, vm, trainw, testw, db)
                mresults.append(item)

            # Initialize KNN models
            for i, model in enumerate(models.knn.get()):
                item = ModelResults(f"KNN{i + 1}", model, vm, trainw, testw, db)
                mresults.append(item)

            # Initialize SVR models
            for i, model in enumerate(models.svr.get()):
                item = ModelResults(f"SVR{i + 1}", model, vm, trainw, testw, db)
                mresults.append(item)

            # Train and test for each split
            for X_train, X_test, y_train, y_test, scaler in utils.getdata(
                trainw, testw, vm, db
            ):
                for mresult in mresults:
                    mresult.model = mresult.model.fit(X_train, y_train)
                    y_hat = mresult.model.predict(X_test)

                    # Rescale the target
                    mresult.add(
                        yhat=utils.rescale(scaler, y_hat),
                        ytrue=utils.rescale(scaler, y_test),
                    )

            # Save all results for this configuration
            for mresult in mresults:
                mresult.save()


# TODO: config file?
vmconfigs = [
    # ("SW", 365, 3),
    ("SW", 730, 3),
    # ("EW", 365, 3),
    ("EW", 730, 3),
]


if __name__ == "__main__":
    # TODO: clean up folder everytime?
    # TODO: see how prophet does with Prediction class
    main()
