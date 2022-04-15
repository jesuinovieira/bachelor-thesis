[![Python 3.8](https://img.shields.io/badge/python-3.8-blue)](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EMB5637: Projeto Integrador II

Project developed for the subject _Integrating Project II_ of the Mechatronics Engineering course at the Federal University of Santa Catarina (UFSC).

## Installation
Using Python 3.8, run the following commands from the project root directory to create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Usage
Once you have everything set up, run the following command to run the project:

```bash
python main.py
```

To run the tests:

```bash
python -m pytest src/tests
```

> Note: tests were not finished.
> Note: the results will differ from the report as the data provided is false.
