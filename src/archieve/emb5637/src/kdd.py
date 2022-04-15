import itertools
import pickle

import pandas as pd

from src import utils
from src.sink import Sink
from src.source import Source
from src.processor import LRProcessor
from src.processor import KNNProcessor
from src.processor import MLPProcessor
from src.processor import SVRProcessor

# --------------------------------------------------------------------------------------
# NOTE: FOCUS ON RESULTS, NOT BEAUTIFUL CODE!
# NOTE: 70% RULE
# --------------------------------------------------------------------------------------

# TODO:
#  - Automatically handle inputs of the next KDD step when disabling some step and
#  assert the necessary stuff is there, otherwise user must run previous step
#  - Improve _getprocessors() function
#  - Create a baseline model: get the best between different ones

# TODO: (processor.py)
#  - Make scaler configurable
#  - Rescale data before or during cross validation?
#  - Don't consider test set when fitting scaler
#  - Grid search or make processor params (which are specific from each method)
#  configurable, currently we have only one model from each method
#  - Use keras for MLP

# TODO: (processor.py, cross validation)
#  https://github.com/alan-turing-institute/sktime/blob/main/sktime/forecasting/model_selection/_split.py
#  https://www.sktime.org/en/v0.6.1/api_reference/modules/auto_generated/sktime.forecasting.model_selection.SlidingWindowSplitter.html
#  https://www.sktime.org/en/v0.6.0/api_reference/modules/auto_generated/sktime.forecasting.model_selection.ExpandingWindowSplitter.html
#  https://towardsdatascience.com/dont-use-k-fold-validation-for-time-series-forecasting-30b724aaea64
#  - window length (wl), forecasting horizon (fh)
#  - Create a class TSCrossValidation, RollingWindow, ExpandingWindow?
#  - Params like vm, trainw, testw seems to be from this new class
#  - Save all cross-validation fold result
#  - Save with timestamp


class KDDPipeline:
    def __init__(self, config):
        print(f"Starting KDD process...")

        self.df = None
        self.config = config

        self._source = Source(
            config["source"]["raw-folder"],
            config["source"]["preprocessed-folder"],
            config["source"]["output-folder"],
        )

        self._processor = []
        self._sink = None

    def _getdf(self):
        dst = self._source.getdst()
        self.df = pd.read_csv(dst)
        self.df.timestamp = pd.to_datetime(
            self.df.timestamp, format="%Y-%m-%d %H:%M:%S"
        )
        self.df = self.df.set_index("timestamp")

    def _getprs(self):
        for processor in self._processor:
            dst = processor.pr.getdst()
            with open(dst, "rb") as f:
                processor.pr = pickle.load(f)

    def select(self, callbacks):
        if not self.config["kdd"]["select"]:
            return

        print(f"[1/5] Selection")
        for callback in callbacks:
            self._source.select(callback)

    def preprocess(self):
        if not self.config["kdd"]["preprocess"]:
            self._getdf()
            return

        print(f"[2/5] Preprocessing")
        # NOTE: immutable, always use self.df.copy()
        self.df = self._source.preprocess()

    def transform(self):
        processors = _getprocessors(self.config["processor"], self.df.copy())
        self._processor.extend(processors)

        if not self.config["kdd"]["transform"]:
            return

        print(f"[3/5] Transformation")
        for processor in self._processor:
            processor.transform()

    def fit(self):
        if not self.config["kdd"]["fit"]:
            self._getprs()
            return

        print(f"[4/5] Data Mining")
        for processor in self._processor:
            print(f"Fitting '{processor._name}' model")
            processor.fit()

    def evaluate(self):
        if not self.config["kdd"]["evaluate"]:
            return

        print(f"[5/5] Evaluation")
        prs = [processor.pr for processor in self._processor]
        self._sink = Sink(prs)
        self._sink.evaluate()


def _getprocessors(config, df):
    def _getprocessor(method, fs, dbname, vm, trainw, testw):
        if "LR" in method:
            Processor = LRProcessor
        elif "KNN" in method:
            Processor = KNNProcessor
        elif "SVR" in method:
            Processor = SVRProcessor
        elif "MLP" in method:
            Processor = MLPProcessor
        else:
            raise AssertionError(f"Method '{method} not supported'")

        return Processor(method, fs, dbname, vm, trainw, testw)

    def _getsubset(innerdf, subset):
        # Subsets
        # DS1: complete database
        # DS2: meteorological attributes
        # DS3: attributes derived from timestamp
        # DS4: holidays and school recess attributes
        attributes = utils.fs2attributes[subset]

        innerdf = innerdf.drop(columns=[col for col in df if col not in attributes])
        innerdf = innerdf[attributes]

        return innerdf.copy()

    tmp = config.values()
    tmp = list(tmp)

    # TODO: this procedure assumes an specific order in the dictionary
    processors = []
    counter = {}
    for item in itertools.product(*tmp):
        method = item[0]
        if method not in counter.keys():
            counter[method] = 0

        counter[method] += 1
        method = f"{item[0]}{counter[method]}"

        params = {
            "method": method,
            "fs": _getsubset(df, item[1]),
            "dbname": item[1],
            "vm": item[2],
            "trainw": item[3],
            "testw": item[4],
        }

        processor = _getprocessor(**params)
        processors.append(processor)

    return processors
