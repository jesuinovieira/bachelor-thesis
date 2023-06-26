import dateutil.parser
import calplot
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# TODO: sex matplotlib text to black!
# TODO: latex and seaborn (googlit)

# Set matplotlib runtime configuration
# --------------------------------------------------------------------------------------
# http://aeturrell.com/2018/01/31/publication-quality-plots-in-python/
# https://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html#Setting-Font-Sizes
# https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-style-sheets
# print(rcParams.keys())

# NOTE: different palette
#  https://stackoverflow.com/questions/46148193/how-to-set-default-matplotlib-axis-colour-cycle
#  Seaborn in fact has six variations of matplotlib’s palette, called deep, muted,
#  pastel, bright, dark, and colorblind. These span a range of average luminance and
#  saturation values: https://seaborn.pydata.org/tutorial/color_palettes.html

# Computer Modern Sanf Serif
# https://seaborn.pydata.org/generated/seaborn.axes_style.html#seaborn.axes_style

# TODO: https://matplotlib.org/stable/users/explain/backends.html

# TODO:
#  1. Set sns grid manually (color)
#  2. Set palette (see note above)
#  3. Maybe let even the black from seaborn in text..
# sns.set_theme(style="whitegrid", palette="pastel")
sns.set_theme()
sns.set_style("whitegrid")
palette = sns.color_palette("colorblind")

SIZE = 8
COLOR = "black"
params = {
    "backend": "ps",
    # "backend": "Agg",

    "axes.titlesize": SIZE,
    "axes.labelsize": SIZE,
    "font.size": SIZE,
    # "text.fontsize": SIZE,
    "legend.fontsize": SIZE,
    "xtick.labelsize": SIZE,
    "ytick.labelsize": SIZE,
    "text.usetex": True,
    "font.family": "serif",
    "text.color": COLOR,
    "axes.labelcolor": COLOR,
    "xtick.color": COLOR,
    "ytick.color": COLOR,
}

rcParams.update(params)

plt.rc("font", size=SIZE)          # controls default text sizes
plt.rc("axes", titlesize=SIZE)     # fontsize of the axes title
plt.rc("axes", labelsize=SIZE)     # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE)    # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE)    # fontsize of the tick labels
plt.rc("legend", fontsize=SIZE)    # legend fontsize
plt.rc("figure", titlesize=SIZE)   # fontsize of the figure title

# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

from matplotlib import font_manager
ticksfont = font_manager.FontProperties(
    family="sans-serif", style="normal", size=10,
    # weight="normal", stretch='normal'
)

# LaTex
# --------------------------------------------------------------------------------------
# The column width is: 455.24411pt
# The text width is: 455.24411pt
# The text height is: 702.78308pt
#
# The paper width is: 597.50787pt
# The paper height is: 845.04684pt

# LaTex
# \message{The column width is: \the\columnwidth}
# \message{The paper width is: \the\paperwidth}
# \message{The paper height is: \the\paperheight}
# \message{The text height is: \the\textheight}
# \message{The text width is: \the\textwidth}

textwidth = 455.24411  # Value given by Latex
textheigth = 702.78308  # Value given by Latex


import matplotlib.pyplot as plt
import matplotlib.ticker


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def setup(xstart=None, xend=None, nrows=2, xfmt=mdates.DateFormatter("%y.%m.%d")):
    """Organize plot style and create matplotlib figure and axis.

    :param xstart: left x axis limit
    :param xend: right x axis limit
    :param nrows: number of rows of the subplot grid
    :param xfmt: formatter of the x axis major ticker
    :return: a tuple with matplotlib figure and axis
    """
    # plt.style.use("ggplot")
    # sns.set_style("whitegrid")

    fig, axs = plt.subplots(nrows)

    # Check if it's an iterable since nrows can be one and axs will be a single object
    for axis in axs if hasattr(axs, "__iter__") else [axs]:
        if xfmt:
            axis.xaxis.set_major_formatter(xfmt)
        if xstart and xend:
            axis.set_xlim(xstart, xend)

    return fig, axs


def save(pdf, filename):
    if not pdf:
        # TODO: save directly and use tight
        # plt.savefig(filename)
        with PdfPages(filename) as pdf:
            wrapup(pdf)
    else:
        wrapup(pdf)


def wrapup(pdf=None, show=False):
    """Finalize current figure. It will clear and close after show and/or save it.

    bbox: https://stackoverflow.com/a/11847260/14113878

    :param pdf: matplotlib PdfPages object to save current figure
    :param show: display current figure
    :param bbox:
    """
    if pdf:
        pdf.savefig(bbox_inches="tight")
    if show:
        plt.show()

    plt.clf()
    plt.close("all")


def get_figsize(columnwidth, wf=0.5, hf=(5. ** 0.5 - 1.0) / 2.0):
    """Parameters:
    - wf [float]: width fraction in columnwidth units
    - hf [float]: height fraction in columnwidth units. Set by default to golden ratio.
    - columnwidth [float]: width of the column in latex. Get this from LaTeX using the
    follwoing command: \showthe\columnwidth

    Returns: [fig_width, fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth * wf
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # Width in inches
    fig_height = fig_width * hf  # Height in inches

    return [fig_width, fig_height]


def zoomin(y, xrange, yrange, axs):
    """
    >>> x1 = df.index[len(df) // 2 - 65]
    >>> x2 = df.index[len(df) // 2 + 65]
    >>> y1 = df[column][len(df) // 2] - 1.5
    >>> y2 = df[column][len(df) // 2] + 1.5
    >>> plot.zoomin(df[column], (x1, x2), (y1, y2), axs)
    """

    axins = zoomed_inset_axes(axs, zoom=2, loc="upper right")
    axins.plot(y)
    axins.set_xlim(*xrange)
    axins.set_ylim(*yrange)

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    mark_inset(axs, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.draw()


# ======================================================================================


def yearly(df, columns, years, pdf):
    # Plot each year
    for year in years:
        prox = str(int(year) + 1)
        tmp = df[df.index > dateutil.parser.parse(f"{year}-01-01")]
        tmp = tmp[tmp.index < dateutil.parser.parse(f"{prox}-01-01")]

        fig, axs = setup(nrows=1)
        tmp.plot(
            kind="line",
            style=".-",
            title=f"{year}",
            y=columns,
            use_index=True,
            ax=axs,
        )

        wrapup(pdf, False)

    # Plot all years
    fig, axs = setup(nrows=1)
    df.plot(
        kind="line",
        style=".-",
        title=f"{years[0]} to {years[-1]}",
        y=columns,
        use_index=True,
        ax=axs,
    )
    wrapup(pdf, False)


def correlation(df, method, pdf, xticks=False):
    corr = df.corr(method).sort_values("water_produced", ascending=True)
    corr = corr.reindex(corr.index, axis=1)

    corrmatrix(corr, f"{method.capitalize()} Correlation\n", pdf, xticks)

    # df = df.dropna()
    #
    # g = sns.pairplot(df, hue="dayofweek", diag_kind="kde")
    # for ax in g.axes.flatten():
    #     ax.set_xlabel(ax.get_xlabel(), rotation=45)
    #     ax.set_ylabel(ax.get_ylabel(), rotation=45)
    #     ax.yaxis.get_label().set_horizontalalignment("right")
    #
    # plt.yticks(rotation=45)
    # plot.wrapup(pdf, False, "tight")
    #
    # g = sns.pairplot(df, diag_kind="kde")
    # for ax in g.axes.flatten():
    #     ax.set_xlabel(ax.get_xlabel(), rotation=90)
    #     ax.set_ylabel(ax.get_ylabel(), rotation=0)
    #     ax.yaxis.get_label().set_horizontalalignment("right")
    #
    # plt.yticks(rotation=45)
    # plt.title("Pair Plot")
    # plot.wrapup(pdf, False, "tight")


def pbc(df, continuous, pdf, xticks=False):
    # Assume all other columns besides from `continuous` are dichotomous (aka binary)
    # len(df.columns)
    corr = pd.DataFrame(index=df.columns, columns=df.columns)

    y = df[continuous]
    for name, values in df.iteritems():
        if name == continuous:
            continue

        x = values
        # x: binary variable (boolean)
        # y: continuous variable
        coef, p = scipy.stats.pointbiserialr(x, y)

        corr.at[name, continuous] = coef

    for col in corr.columns:
        corr[col] = pd.to_numeric(corr[col], errors="coerce")

    corr = corr.sort_values(by="water_produced", ascending=True)
    corr = corr.reindex(corr.index, axis=1)

    corrmatrix(corr, f"PBS Correlation\n", pdf, xticks)


def corrmatrix(corr, title, pdf, xticks=False):
    sns.heatmap(
        corr,
        square=True,
        fmt=".2f",
        cmap=sns.color_palette("vlag", as_cmap=True),
        annot=True,
        mask=corr.isnull(),  # annot_kws={"size": 8}
    )

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    if not xticks:
        plt.xticks([], [])
    plt.title(title)

    # wrapup(pdf, False, "tight")


def calhm(dates, title, pdf):
    calplot.calplot(
        dates,
        cmap="inferno",
        colorbar=False,
        linewidth=3,
        edgecolor="gray",
        figsize=rcParams["figure.figsize"],
        suptitle=f"\n{title}",
    )

    wrapup(pdf, False)
