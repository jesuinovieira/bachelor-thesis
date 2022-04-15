import logging
import sys


def setup_logger(name, logfile):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    fhandler = logging.FileHandler(logfile)
    shandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fhandler.setFormatter(formatter)
    # formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    shandler.setFormatter(formatter)

    params = {"level": logging.DEBUG, "handlers": [fhandler, shandler]}
    logging.basicConfig(**params)

    fhandler.setLevel(logging.INFO)
    shandler.setLevel(logging.DEBUG)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.WARNING)
    # logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logging.getLogger(name)
