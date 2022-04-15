from sklearn.svm import SVR


def get():
    return [SVR(kernel="rbf", C=1, epsilon=0.1)]
