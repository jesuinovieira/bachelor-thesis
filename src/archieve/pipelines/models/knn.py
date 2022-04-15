from sklearn.neighbors import KNeighborsRegressor


def get():
    return [
        KNeighborsRegressor(n_neighbors=3, weights="distance", metric="minkowski", p=2),
        # KNeighborsRegressor(
        #     n_neighbors=5, weights="distance", metric="minkowski", p=2
        # ),
        KNeighborsRegressor(n_neighbors=7, weights="distance", metric="minkowski", p=2),
        # KNeighborsRegressor(
        #     n_neighbors=9, weights="distance", metric="minkowski", p=2
        # ),
        KNeighborsRegressor(
            n_neighbors=11, weights="distance", metric="minkowski", p=2
        ),
    ]
