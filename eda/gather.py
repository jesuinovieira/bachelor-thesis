import os

import pandas as pd


def _getcols(areas, city):
    cols = ["timestamp"]
    for area in areas:
        cols.append(f"consumed-{area}")
        cols.append(f"fr-{area}")
    cols.append(f"consumed-{city}")

    return cols


def _createdfs(df, areas, city):
    # Consumed: volume consumed in m3
    consumed = f"consumed-{city}"
    dftotal = pd.DataFrame({"timestamp": df.timestamp, f"{consumed}": df[consumed]})
    dfareas = {}

    for area in areas:
        data = {
            "timestamp": df.timestamp,
            "consumed": df[f"consumed-{area}"],
            "fr": df[f"fr-{area}"]
        }

        dfareas[f"{area}"] = pd.DataFrame(data)

    return dftotal, dfareas


def _extenddfs(df, areas, city, dftotal, dfareas):
    _dftotal, _dfareas = _createdfs(df, areas, city)

    dftotal = dftotal.append(_dftotal)
    for area in areas:
        dfareas[area] = dfareas[area].append(_dfareas[area])

    return dftotal, dfareas


def _createcolumn(city, dftotal, dfareas):
    length = len(dftotal)
    new = []

    for i in range(length):
        values = [item.iloc[i].consumed for item in dfareas.values()]
        summed = sum(values)
        new.append(summed)

    dftotal[f"consumed-sum-{city}"] = new

    return dftotal


def _validate(total, pl, canoas, ipanema, shangrila, atami):
    length = len(total)
    lst = [pl, canoas, ipanema, shangrila, atami]

    for i in range(length):
        summedup = sum([item.iloc[i].consumed for item in lst])
        consumed = total.iloc[i].consumed

        if summedup != consumed:
            print(f"{summedup} != {consumed}")
            pass


def vpufpr(src, cols, city):
    # Properly ready the data
    df = pd.read_excel(src, sheet_name=None, skiprows=2, usecols=cols)
    planilha = "Planilha2" if city == "gtba" else "Planilha1"
    if city == "pp":
        columns = {"Produzido (m3).1": f"produced-{city}", "Data.1": "timestamp"}
    else:
        columns = {"Produzido (m3)": f"produced-{city}", "Data": "timestamp"}

    df = df[planilha]
    df = df.drop(df.tail(1).index)
    df = df.rename(columns=columns)

    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
    df = df.set_index("timestamp")

    dst = f"vp-ufpr-{city.lower()}.csv"
    return df, dst


def ufprdiario(srcs, areas, city):
    n = 1 + 3 * len(areas) + 1
    usecols = [i for i in range(0, n) if i % 3 != 0 or i == 0]

    dftotal, dfareas = None, {}
    cols = _getcols(areas, city)

    for src in srcs:
        months = pd.read_excel(src, sheet_name=None, skiprows=3, usecols=usecols)

        for month, df in months.items():
            df.columns = cols
            df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")

            if dftotal is None:
                dftotal, dfareas = _createdfs(df, areas, city)
            else:
                dftotal, dfareas = _extenddfs(df, areas, city, dftotal, dfareas)

    dftotal = dftotal.set_index("timestamp")
    for area in areas:
        dfareas[area] = dfareas[area].set_index("timestamp")

    # Some lines are duplicated, probably due to bad formatting of the .xlsx file.
    # Fortunately, the correct value is always the last one
    # print(dftotal[dftotal.index.duplicated(keep=False)])
    dftotal = dftotal[~dftotal.index.duplicated(keep="last")]
    for area in areas:
        dfareas[area] = dfareas[area][~dfareas[area].index.duplicated(keep="last")]

    # Create a column that is the sum of the volume of all areas
    dftotal = _createcolumn(city, dftotal, dfareas)

    # _validate(total, pl, canoas, ipanema, shangrila, atami)

    dst = f"ufpr-diario-{city.lower()}.csv"
    return dftotal, dst


def main(basepath="eda/data"):
    dflist = []
    external = os.path.join(basepath, "external")
    raw = os.path.join(basepath, "raw")

    if not os.path.isdir(raw):
        os.makedirs(raw)

    # vp-ufpr
    # ----------------------------------------------------------------------------------

    # @matinhos
    src = os.path.join(external, "VP UFPR.xlsx")

    df, dst = vpufpr(src, (5, 7), "matinhos")
    dst = os.path.join(raw, dst)
    df.to_csv(dst, index=True, index_label="timestamp")
    dflist.append(df)

    # @pp
    src = os.path.join(external, "VP UFPR.xlsx")

    df, dst = vpufpr(src, (10, 12), "pp")
    dst = os.path.join(raw, dst)
    df.to_csv(dst, index=True, index_label="timestamp")
    dflist.append(df)

    # @gtba
    src = os.path.join(external, "VP UFPR GTBA.xlsx")

    df, dst = vpufpr(src, (6, 8), "gtba")
    dst = os.path.join(raw, dst)
    df.to_csv(dst, index=True, index_label="timestamp")
    dflist.append(df)

    # ufpr-diario
    # ----------------------------------------------------------------------------------

    # @matinhos
    # ?

    # @pp
    src = [
        os.path.join(basepath, "external/UFPR - DIµRIO 2016.xlsx"),
        os.path.join(basepath, "external/UFPR - DIµRIO 2017.xlsx"),
        os.path.join(basepath, "external/UFPR - DIµRIO 2018.xlsx"),
    ]

    areas = ["eta-pl", "canoas", "ipanema", "shangri-la", "atami"]
    df, dst = ufprdiario(src, areas, "pp")
    dst = os.path.join(raw, dst)
    df.to_csv(dst, index=True, index_label="timestamp")
    dflist.append(df)

    # @gbta
    src = [
        os.path.join(basepath, "external/UFPR - DI╡RIO 2016 GTBA.xlsx"),
        os.path.join(basepath, "external/UFPR - DI╡RIO 2017 GTBA.xlsx"),
        os.path.join(basepath, "external/UFPR - DI╡RIO 2018 GTBA.xlsx"),
        os.path.join(basepath, "external/UFPR - DI╡RIO 2019 GTBA.xlsx"),
    ]

    areas = ["coroados", "brejatuba", "aeroporto", "central"]
    df, dst = ufprdiario(src, areas, "gtba")
    dst = os.path.join(raw, dst)
    df.to_csv(dst, index=True, index_label="timestamp")
    dflist.append(df)

    # Merge gathered data into a single dataframe
    # ----------------------------------------------------------------------------------

    df = pd.concat(dflist, axis=1)
    dst = os.path.join(raw, "merged.csv")
    df.to_csv(dst, index=True, index_label="timestamp")


if __name__ == "__main__":
    main()
