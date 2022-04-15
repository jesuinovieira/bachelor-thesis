from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

import utils


trainw, testw = 30, 7
method = "sliding-window"

results = {"R2": [], "RMSE": [], "MAE": [], "MSE": []}
for X_train, X_test, y_train, y_test in utils.getdata(trainw, testw, method):

    # TODO: what about the last split? Is not "complete"
    if len(y_test) != testw:
        continue

    model = LinearRegression()
    model = model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    # R2 = r2_score(y_test, y_hat)
    # MAE = mean_absolute_error(y_test, y_hat)
    # RMSE = mean_squared_error(y_test, y_hat, squared=False)
    # MSE = mean_squared_error(y_test, y_hat, squared=True)
    # # squared: if True returns MSE value, if False returns RMSE value.
    #
    # results["R2"].append(R2)
    # results["MAE"].append(MAE)
    # results["RMSE"].append(RMSE)
    # results["MSE"].append(MSE)

    # TODO: which results should I save for each split?
    # TODO: which results should I save for the general results? Should I?

    # TODO: DOE blocked cross-validation
    # TODO: DOE window size and configurations

    # TODO: which plots do I want to do? Which data should I save for that?
    # TODO: save X_train, X_test, y_train and y_test for each split?
    # TODO: save scaler for each split?

    # TODO: modularize pipeline for any model

# print(
#     f"R2: {round(np.mean(results['R2']).item(), 4)}\n"
#     f"MAE: {round(np.mean(results['MAE']).item(), 4)}\n"
#     f"RMSE: {round(np.mean(results['RMSE']).item(), 4)}\n"
#     f"MSE: {round(np.mean(results['MSE']).item(), 4)}"
# )

# TODO: what about the last split? Is not "complete"

# TODO: no trabalho do Angelo, apenas uma configuração de janela não ancorada e ancorada
#  foram utilizadas, correto? Não há comparações baseadas no tamanho das janelas

# Save y_true and y_hat for each split (later any of the metrics can be calculated)
# Later on, calculate the metric average and the standard deviation

# Sliding window
#   CFG1, ..., CFGN
#       DS1, ..., DSN
#           M1, ..., MN
# Expanding window
#   CFG1, ..., CFGN
#       DS1, ..., DSN
#           M1, ..., MN

# "Os resultados foram obtidos a partir de um total de 33 configurações (10 MLPs,
# 15 SVRs, 5 kNNs, 2 Árvores de Decisão, e 1 Regressão Linear)" (Angelo)

# Tabela 6.3 – Melhores resultados obtidos com cada modelo e conjunto de entrada.

# ======================================================================================

# Preparing data for linear regression
# https://machinelearningmastery.com/linear-regression-for-machine-learning/
# ISLR

# Research: linear regression rules of thumb

# Regularization (lasso, ridge)
# https://chrisalbon.com/code/machine_learning/linear_regression/ridge_regression/
# Basis expansion?
# from sklearn.pipeline import Pipeline
