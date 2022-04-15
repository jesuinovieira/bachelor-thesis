import json

import pandas as pd
import pytest

import src.kdd
import src.utils

CFGFILE = "src/tests/fixture/config.json"


class SourceMocker:
    def __init__(self):
        self.selected = False
        self.preprocessed = False

    def select(self, callback):
        assert not self.selected
        self.selected = True

    def preprocess(self):
        assert not self.preprocessed
        self.preprocessed = True


class ProcessorMocker:
    def __init__(self):
        self.transformed = False
        self.fitted = False
        self._name = "ProcessorMocker"

    def transform(self):
        assert not self.transformed
        self.transformed = True

    def fit(self):
        assert not self.fitted
        self.fitted = True


def getcfg():
    with open(CFGFILE) as file:
        config = json.load(file)
    return config


def getkdd():
    config = getcfg()
    return src.kdd.KDDPipeline(config=config)


def getemptydf():
    columns = (
        src.utils.fs2attributes["FS1"]
        + src.utils.fs2attributes["FS2"]
        + src.utils.fs2attributes["FS3"]
        + src.utils.fs2attributes["FS4"]
    )
    df = pd.DataFrame(columns=columns)
    return df


def callback():
    return getemptydf(), "empty-df.csv"


@pytest.mark.unit
def test_select_ok():
    kdd = getkdd()
    kdd._source = SourceMocker()

    callbacks = (callback,)
    kdd.select(callbacks)
    assert kdd._source.selected


@pytest.mark.unit
def test_preprocess_ok():
    kdd = getkdd()
    kdd._source = SourceMocker()

    kdd.preprocess()
    assert kdd._source.preprocessed


# TODO: what to do here? List of processors is created inside the method
# @pytest.mark.unit
# def test_transform_ok():
#     pass


@pytest.mark.unit
def test_fit_ok():
    kdd = getkdd()
    kdd._processor = [ProcessorMocker(), ProcessorMocker(), ProcessorMocker()]

    kdd.fit()

    for item in kdd._processor:
        assert item.fitted


@pytest.mark.unit
def test_evaluate_ok():
    # TODO: what to do here? Sink is created inside the method
    pass


@pytest.mark.unit
def test_getprocessors_ok():
    # Create fake df (only with dummie columns, besides the actual ones) fixture
    # Assert FS1, FS2, FS3 and FS4
    df = getemptydf()
    config = getcfg()

    # config["processor"]["methods"] = "LR"
    # config["processor"]["feature-set"] = ["FS1", "FS2", "FS3", "FS4"]

    processors = src.kdd._getprocessors(config["processor"], df)
    assert len(processors) == 8
