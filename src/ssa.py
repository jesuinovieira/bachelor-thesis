# https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition/data#4.-A-Python-Class-for-SSA
import numpy as np


class SSA(object):
    __supported_types = (np.ndarray, list)

    def __init__(self, tseries, L=None, save_mem=True):
        """Decomposes the given time series with a singular-spectrum analysis. Assumes
        the values of the time series are recorded at equal intervals.

        :param tseries: the original time series, in the form of a numpy array or list
        :param L: the window length. Must be an integer 2 <= L <= N/2, where N is the
        length of the time series
        :param save_mem: conserve memory by not retaining the elementary matrices.
        Recommended for long time series with thousands of values. Defaults to True.
        """
        # Type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try numpy array or list.")

        # Checks to save us from ourselves
        self.N = len(tseries)
        self.L = L if L else self.N // 2
        if not 2 <= self.L <= self.N / 2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.timeseries = np.array(tseries)
        self.K = self.N - self.L + 1

        # (1) Embedding: a one-dimensional time series is converted into a higher
        # dimension Hankel matrix, known as the trajectory matrix. The dimension of the
        # trajectory matrix is L x K (rows x columns), where:
        #   L = window length
        #   K = N - L + 1
        self.X = np.array([self.timeseries[i : self.L + i] for i in range(0, self.K)]).T

        # (2) SVD: Decompose the trajectory matrix. SVD is applied to the trajectory
        # matrix and eigenvalues and eigenvectors are found. The triple Ui, Si and Vi is
        # called as ith eigentriple of the SVD
        self.U, self.S, self.V = np.linalg.svd(self.X)

        # Identify the rank of the embedding subspace
        self.d = np.linalg.matrix_rank(self.X)

        # Components
        self.components = np.zeros((self.N, self.d))

        if not save_mem:
            # (3) Eingentriple grouping: split the elementary matrices into several
            # groups and sum the matrices in each group.
            self.elementaryX = np.array(
                [
                    self.S[i] * np.outer(self.U[:, i], self.V[i, :])
                    for i in range(self.d)
                ]
            )

            # (4) Diagonal averaging: by taking the average along the diagonals of each
            # group we get reconstructed components and combining them into one time
            # series we obtain the approximated original one.
            for i in range(self.d):
                # Store them as columns
                revX = self.elementaryX[i, ::-1]
                self.components[:, i] = [
                    revX.diagonal(j).mean()
                    for j in range(-revX.shape[0] + 1, revX.shape[1])
                ]

            self.V = self.V.T
        else:
            for i in range(self.d):
                elementaryX = self.S[i] * np.outer(self.U[:, i], self.V[i, :])
                revX = elementaryX[::-1]
                self.components[:, i] = [
                    revX.diagonal(j).mean()
                    for j in range(-revX.shape[0] + 1, revX.shape[1])
                ]

            # Run with save_mem=False to retain the elementary matrices and V array
            self.elementaryX = None
            self.V = None

        self.wcorr = None

    def get_components(self, n=0):
        """Returns the first n time series components."""
        if self.d == 0:
            return self.timeseries

        n = min(n, self.d) if n > 0 else self.d
        return self.components[:, :n]

    def reconstruct(self, r):
        """Reconstructs the time series from its elementary components, using the given
        indices.

        :param r: an integer, list of integers or slice(n,m) object, representing the
        elementary components to sum
        :return: a numpy array object with the reconstructed time series
        """
        if self.d == 0:
            return self.timeseries

        if isinstance(r, int):
            r = [r]

        return self.components[:, r].sum(axis=1)

    def wcorrelation(self):
        """Calculates the w-correlation matrix for the time series."""
        # Calculate the weights
        w = np.array(
            list(np.arange(self.L) + 1)
            + [self.L] * (self.K - self.L - 1)
            + list(np.arange(self.L) + 1)[::-1]
        )

        def inner(F_i, F_j):
            return w.dot(F_i * F_j)

        # Calculated weighted norms, ||F_i||_w, then invert.
        F_wnorms = np.array(
            [inner(self.components[:, i], self.components[:, i]) for i in range(self.d)]
        )
        F_wnorms = F_wnorms**-0.5

        # Calculate Wcorr
        self.wcorr = np.identity(self.d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                self.wcorr[i, j] = abs(
                    inner(self.components[:, i], self.components[:, j])
                    * F_wnorms[i]
                    * F_wnorms[j]
                )
                self.wcorr[j, i] = self.wcorr[i, j]

    def plotwcorr(self, ax, sns, plt):
        """Plots the w-correlation matrix for the decomposed time series.

        :param plt: matplotlib pyplot
        """
        if self.wcorr is None:
            self.wcorrelation()

        _min = 0
        _max = self.d

        # Define a cmap
        rbg = np.linspace(0, 1, 10)[::-1]
        cmap = [(item, item, item) for item in rbg]
        # cmap[1] = cmap[2] = cmap[3] = cmap[4]
        # cmap[5] = cmap[6] = cmap[7] = cmap[8]

        # cmap = "gray_r"
        # cmap = "YlGnBu"
        # cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # cmap = sns.diverging_palette(220, 20, n=10)

        sns.heatmap(
            self.wcorr, ax=ax, square=True,
            cmap=cmap, linewidths=0.0, rasterized=True, cbar_kws={"shrink": 0.75}
        )

        # plt.xticks(ticks=np.arange(0, 560, 56))

        # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
        #             square=True, linewidths=.5, cbar_kws={"shrink": .5})
        # sns.heatmap(self.wcorr, ax=ax, cmap=cmap)
        # plt.xlabel(r"$\tilde{F}_i$")
        # plt.ylabel(r"$\tilde{F}_j$")
        # plt.colorbar(ax.colorbar, fraction=0.045)
        # ax.colorbar.set_label("$W_{i,j}$")
        # plt.clim(0, 1)

        # For plotting purposes
        # max_rnge = _max - 1
        # plt.xlim(_min - 0.5, max_rnge + 0.5)
        # plt.ylim(max_rnge + 0.5, _min - 0.5)
