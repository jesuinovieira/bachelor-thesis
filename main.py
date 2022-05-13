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


def main(cfgfile):
    with open(cfgfile) as file:
        cfg = json.load(file)

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

    kdd = KDDPipeline(config=cfg, output=outfolder)
    kdd.select(callbacks)
    kdd.preprocess()
    kdd.transform()
    kdd.fit()
    kdd.evaluate()


if __name__ == "__main__":
    cfgfolder = "."
    configs = ["config.json"]

    for config in configs:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        outfolder = os.path.join("output", now)

        config = os.path.join(cfgfolder, config)

        if not os.path.isdir(outfolder):
            os.makedirs(outfolder, exist_ok=True)

        src = config
        dst = os.path.join(outfolder, os.path.basename(src))
        shutil.copyfile(src, dst)

        logfile = os.path.join(outfolder, "output.log")
        logger = setuplogger(logfile)

        start = time.time()
        main(config)
        end = time.time()
        delta = end - start

        den, unit = 1, "seconds"
        if delta >= 60:
            den, unit = 60, "minutes"
        if delta >= 3600:
            den, unit = 3600, "hours"

        logger.info(f"Elapsed time: {round(delta / den, 2)} {unit}")
