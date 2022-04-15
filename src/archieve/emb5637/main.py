import json
from functools import partial

import callback
import manual
from src import KDDPipeline

CFGFILE = "config.json"


if __name__ == "__main__":
    with open(CFGFILE) as file:
        config = json.load(file)

    kdd = KDDPipeline(config=config)

    cities = [("GUARATUBA", "PR"), ("CURITIBA", "PR"), ("JOINVILLE", "SC")]

    callbacks = (
        partial(callback.readfakedata, "data/external/fake-vp-ufpr-gtba.csv"),
        partial(callback.readfakedata, "data/external/fake-guaratuba.csv"),
        partial(callback.holidays, "GUARATUBA", "PR"),
        partial(callback.holidays, "CURITIBA", "PR"),
        partial(callback.holidays, "JOINVILLE", "SC"),
        partial(callback.manual, manual.CARNIVAL, "is_carnival", "carnival.csv"),
        partial(
            callback.manual,
            manual.SCHOOL_RECESS_PR,
            "is_school_recess_pr",
            "school-recess-pr.csv",
        ),
    )

    kdd.select(callbacks)
    kdd.preprocess()
    kdd.transform()
    kdd.fit()
    kdd.evaluate()
