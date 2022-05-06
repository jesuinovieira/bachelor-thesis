import json
import time
from functools import partial

import callback
import manual
from src import KDDPipeline

CFGFILE = "config.json"


def main():
    with open(CFGFILE) as file:
        config = json.load(file)

    callbacks = (
        partial(callback.waterproduced, "data/external/VP UFPR GTBA.xlsx"),
        partial(callback.meteorological, "data/external/Guaratuba.xlsx"),
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

    kdd = KDDPipeline(config=config)
    kdd.select(callbacks)
    kdd.preprocess()
    kdd.transform()
    kdd.fit()
    kdd.evaluate()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    delta = end - start

    den, unit = 1, "seconds"
    if delta >= 60:
        den, unit = 60, "minutes"
    if delta >= 3600:
        den, unit = 3600, "hours"

    print(
        f"\n"
        f"Elapsed time: {round(delta / den, 2)} {unit}"
    )
