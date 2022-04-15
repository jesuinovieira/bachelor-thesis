import dateutil.parser

import calplot
import seaborn as sns
import matplotlib.dates as mdates
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# Set matplotlib runtime configuration
DPI = 100
rcParams["figure.autolayout"] = True
rcParams["figure.figsize"] = (1920 / DPI, 986 / DPI)
rcParams["font.family"] = "monospace"

# Colors
DB = "darkblue"
RB = "royalblue"
DO = "darkorange"
CB = "cornflowerblue"

# Basic params
params = {"linestyle": "-", "linewidth": 1.0, "marker": "o", "markersize": 2.0}


def setup(xstart=None, xend=None, nrows=2, xfmt=mdates.DateFormatter("%y.%m.%d")):
    """Organize plot style and create matplotlib figure and axis.

    :param xstart: left x axis limit
    :param xend: right x axis limit
    :param nrows: number of rows of the subplot grid
    :param xfmt: formatter of the x axis major ticker
    :return: a tuple with matplotlib figure and axis
    """
    plt.style.use("ggplot")
    fig, axs = plt.subplots(nrows)

    # Check if it's an iterable since nrows can be one and axs will be a single object
    for axis in axs if hasattr(axs, "__iter__") else [axs]:
        if xfmt:
            axis.xaxis.set_major_formatter(xfmt)
        if xstart and xend:
            axis.set_xlim(xstart, xend)

    return fig, axs


def wrapup(pdf=None, show=False, bbox=None):
    """Finalize current figure. It will clear and close after show and/or save it.

    :param pdf: matplotlib PdfPages object to save current figure
    :param show: display current figure
    :param bbox:
    """
    if pdf:
        pdf.savefig(bbox_inches=bbox)
    if show:
        plt.show()

    plt.clf()
    plt.close("all")


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

    wrapup(pdf, False, "tight")


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
