import pandas as pd


def calendar():
    FILENAME = "data/raw/Feriados CURITIBA.csv"
    # FILENAME = "data/raw/Feriados GUARATUBA.csv"
    # FILENAME = "data/raw/Feriados JOINVILLE.csv"
    YEAR = 2016

    df = pd.read_csv(FILENAME)

    # Drop useless columns
    df = df.drop(
        # ["link", "description", "type_code", "raw_description", "name", "type"],
        ["link", "description", "type_code", "raw_description"],
        axis=1,
    )
    df = df.rename(columns={"date": "timestamp"})
    df.timestamp = pd.to_datetime(df.timestamp, format="%d/%m/%Y")
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d")

    tmp = df[df.timestamp.dt.year == YEAR]

    print(tmp[tmp.type == "Facultativo"])
    print()
    print(tmp[tmp.type == "Dia Convencional"])


def main():
    calendar()


if __name__ == "__main__":
    main()
