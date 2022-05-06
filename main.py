import datetime
import json
import logging
import os.path
import shutil
import sys
import time
from functools import partial

import callback
import manual
from src import KDDPipeline

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
CFGFILE = "config.json"
OUTFOLDER = os.path.join("output", now)


def setuplogger(file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    fhandler = logging.FileHandler(file)
    shandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fhandler.setFormatter(formatter)

    formatter = logging.Formatter("%(message)s")
    shandler.setFormatter(formatter)

    params = {"level": logging.INFO, "handlers": [fhandler, shandler]}
    logging.basicConfig(**params)

    fhandler.setLevel(logging.INFO)
    shandler.setLevel(logging.INFO)

    return logging.getLogger(__name__)


def main():
    src = CFGFILE
    dst = os.path.join(OUTFOLDER, CFGFILE)
    shutil.copyfile(src, dst)

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

    kdd = KDDPipeline(config=config, output=OUTFOLDER)
    kdd.select(callbacks)
    kdd.preprocess()
    kdd.transform()
    kdd.fit()
    kdd.evaluate()


if __name__ == "__main__":
    if not os.path.isdir(OUTFOLDER):
        os.makedirs(OUTFOLDER, exist_ok=True)

    logfile = os.path.join(OUTFOLDER, "output.log")
    logger = setuplogger(logfile)

    start = time.time()
    main()
    end = time.time()
    delta = end - start

    den, unit = 1, "seconds"
    if delta >= 60:
        den, unit = 60, "minutes"
    if delta >= 3600:
        den, unit = 3600, "hours"

    logger.info(f"Elapsed time: {round(delta / den, 2)} {unit}")
