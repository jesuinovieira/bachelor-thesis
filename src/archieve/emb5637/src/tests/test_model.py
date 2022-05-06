import json
import os
import pickle

import pandas as pd

CFGFILE = "src/tests/fixture/config.json"


def getcfg():
    with open(CFGFILE) as file:
        config = json.load(file)
    return config


# Pre-train tests
# ======================================================================================


def test_output_shape():
    # All regressors output a single number
    pass


def test_output_range():
    # Theoretically from -inf to +inf for regression problems
    pass


def test_gradient_step():
    # Out of my control, scikit-learn does the training and don't share information
    # about each gradient step
    pass


def test_dataset():
    # ?
    pass


def test_label_leakage():
    # "When the information you want to predict is directly or indirectly present in
    # your training dataset"
    #
    # Can't automatically do that, but throughout the project this was verified,
    # especially during the transformation (MinMaxScaler) of the data
    pass


# Post-train tests
# ======================================================================================

# Invariance tests for regression?
# --------------------------------------------------------------------------------------

# Directional expectation tests: define a set of perturbations to the input which should
# have a predictable effect on the model output
# --------------------------------------------------------------------------------------


def getprs(folder="output/processor"):
    prs = []
    for file in os.listdir(folder):
        if file.endswith(".pickle"):
            with open(file, "rb") as f:
                pr = pickle.load(f)
                prs.append(pr)
    pass


def getpredictions(prs, id):
    for pr in prs:
        if pr.id == id:
            data = {"yhat": pr.yhat, "ytrue": pr.ytrue}
            df = pd.DataFrame(data=data, index=pr.timestamp)
            df.index = pd.to_datetime(df.index)
            return df


def getbestmodel(df):
    methods = ["LR", "KNN", "SVR", "MLP"]
    for method in methods:
        # Select the model with best perfomance based on some metric
        filtered = df.filter(like=method, axis="index")
        filtered = filtered[filtered.r2 == filtered.r2.max()]

        if filtered.empty:
            continue

        yield method, getpredictions(filtered.index[0])


def test_temperature_mean_perturbation():
    pass


def test_radiation_mean_perturbation():
    pass


def test_relative_humidity_perturbation():
    pass


def test_is_school_recess_pr_perturbation():
    pass


def test_is_holiday_ctba_jve_perturbation():
    pass


def test_is_carnival_perturbation():
    pass


# Minimum functionality tests: data unit tests allow us to quantify model performance
# for specific cases found in your data
# --------------------------------------------------------------------------------------


def test_per_season():
    pass


def test_per_dayofweek():
    pass
