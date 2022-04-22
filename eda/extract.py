def main():
    import os
    import zipfile

    dst = "eda/data/external/test"

    # Extract data
    src = "eda/data/external/fwddadossanepar.zip"
    with zipfile.ZipFile(src, "r") as f:
        f.extractall(dst)

    src = "eda/data/external/fwddadossimeparesanepar.zip"
    with zipfile.ZipFile(src, "r") as f:
        f.extractall(dst)

    src = "eda/data/external/test/UFPR - GTBA.zip"
    with zipfile.ZipFile(src, "r") as f:
        f.extractall(dst)

    # Remove unused files
    for item in os.listdir(dst):
        if not item.endswith(".xlsx") and not item.endswith(".xls"):
            file = os.path.join(dst, item)
            os.remove(file)

    # Rename files
    old = os.path.join(dst, "Guaratuba.xlsx")
    new = os.path.join(dst, "simepar-gtba.xlsx")
    os.rename(old, new)

    for year in ["2016", "2017", "2018", "2019"]:
        old = os.path.join(dst, f"UFPR - DI╡RIO {year} GTBA.xlsx")
        new = os.path.join(dst, f"consumed-gtba-{year}.xlsx")
        os.rename(old, new)

    for year in ["2016", "2017", "2018"]:
        old = os.path.join(dst, f"UFPR - DIµRIO {year}.xlsx")
        new = os.path.join(dst, f"consumed-pp-{year}.xlsx")
        os.rename(old, new)


if __name__ == "__main__":
    print(f"Not sure if it's working...")
    # main()
