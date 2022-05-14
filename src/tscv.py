import math

# TODO: make code reusable (SW and EW are very similar)


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
