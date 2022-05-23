import itertools
import logging
import os
import pickle

import pandas as pd

from src.sink import Sink
from src.source import Source
from src.processor import LRProcessor
from src.processor import KNNProcessor
from src.processor import MLPProcessor
from src.processor import SVRProcessor

logger = logging.getLogger(__name__)


class KDDPipeline:
    def __init__(self, config, output):
        logger.info(f"Starting KDD process...")

        self.df = None
        self.config = config
        self.output = output

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
        # FIXME: deprecated
        logger.error(f"Deprecated")

        for processor in self._processor:
            dst = processor.pr.getdst()
            with open(dst, "rb") as f:
                processor.pr = pickle.load(f)

    def select(self, callbacks):
        if not self.config["kdd"]["select"]:
            return

        logger.info(f"[1/5] Selection")
        for callback in callbacks:
            self._source.select(callback)

    def preprocess(self):
        if not self.config["kdd"]["preprocess"]:
            self._getdf()
            return

        logger.info(f"[2/5] Preprocessing")
        # NOTE: immutable, always use self.df.copy()
        self.df = self._source.preprocess()

    def transform(self):
        output = os.path.join(self.output, "processor")
        # NOTE: _getprocessors creates a deep copy of the database for each model
        processors = _getprocessors(self.config["processor"], self.df, output)
        self._processor.extend(processors)

        if not self.config["kdd"]["transform"]:
            return

        logger.info(f"[3/5] Transformation")
        for processor in self._processor:
            processor.transform()

    def fit(self):
        if not self.config["kdd"]["fit"]:
            self._getprs()
            return

        logger.info(f"[4/5] Data Mining")
        for processor in self._processor:
            logger.info(f"Fitting '{processor.pr.id}' model")
            processor.fit()

    def evaluate(self):
        if not self.config["kdd"]["evaluate"]:
            return

        logger.info(f"[5/5] Evaluation")

        output = os.path.join(self.output, "sink")
        if not os.path.isdir(output):
            os.makedirs(output, exist_ok=True)

        prs = [processor.pr for processor in self._processor]
        self._sink = Sink(prs, path=output)
        self._sink.evaluate()


def _getprocessors(config, df, output):
    Processor = dict(
        LR=LRProcessor, KNN=KNNProcessor, SVR=SVRProcessor, MLP=MLPProcessor
    )

    if not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)

    tmp = config.values()
    tmp = list(tmp)

    processors = []
    counter = {}

    # FIXME: this procedure assumes an specific order in the dict, order it!
    for item in itertools.product(*tmp):
        method = item[0]
        if method not in counter.keys():
            counter[method] = 0

        counter[method] += 1
        params = dict(
            id=f"{item[0]}{counter[method]}",
            df=df.copy(deep=True),
            backtest=item[1],
            output=output,
        )

        processor = Processor[method](**params)
        processors.append(processor)

    return processors
