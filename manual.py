import pandas as pd

# http://www.gestaoescolar.diaadia.pr.gov.br/modules/conteudo/conteudo.php?conteudo=27
CARNIVAL = (
    # http://g1.globo.com/carnaval/2016/noticia/2015/10/carnaval-2016-veja-datas.html
    pd.date_range(start="2016-02-05", end="2016-02-10").to_pydatetime().tolist()
    # https://g1.globo.com/carnaval/2017/noticia/carnaval-2017-veja-datas.ghtml
    + pd.date_range(start="2017-02-24", end="2017-03-01").to_pydatetime().tolist()
    # https://g1.globo.com/carnaval/2018/noticia/carnaval-2018-veja-datas.ghtml
    + pd.date_range(start="2018-02-09", end="2018-02-14").to_pydatetime().tolist()
    # https://g1.globo.com/carnaval/2019/noticia/carnaval-2019-veja-datas.ghtml
    + pd.date_range(start="2019-03-01", end="2019-03-06").to_pydatetime().tolist()
)

# http://www.gestaoescolar.diaadia.pr.gov.br/modules/conteudo/conteudo.php?conteudo=27
SCHOOL_RECESS_PR = (
    pd.date_range(start="2016-01-01", end="2016-02-28").to_pydatetime().tolist()
    + pd.date_range(start="2016-04-22", end="2016-04-22").to_pydatetime().tolist()
    + pd.date_range(start="2016-05-27", end="2016-05-27").to_pydatetime().tolist()
    + pd.date_range(start="2016-07-16", end="2016-07-31").to_pydatetime().tolist()
    + pd.date_range(start="2016-11-14", end="2016-11-14").to_pydatetime().tolist()
    + pd.date_range(start="2016-12-22", end="2016-12-31").to_pydatetime().tolist()
    + pd.date_range(start="2017-01-01", end="2017-02-14").to_pydatetime().tolist()
    + pd.date_range(start="2017-02-27", end="2017-02-27").to_pydatetime().tolist()
    + pd.date_range(start="2017-03-01", end="2017-03-01").to_pydatetime().tolist()
    + pd.date_range(start="2017-03-06", end="2017-03-06").to_pydatetime().tolist()
    + pd.date_range(start="2017-05-24", end="2017-05-24").to_pydatetime().tolist()
    + pd.date_range(start="2017-06-02", end="2017-06-02").to_pydatetime().tolist()
    + pd.date_range(start="2017-06-16", end="2017-06-16").to_pydatetime().tolist()
    + pd.date_range(start="2017-07-15", end="2017-07-25").to_pydatetime().tolist()
    + pd.date_range(start="2017-09-08", end="2017-09-08").to_pydatetime().tolist()
    + pd.date_range(start="2017-09-26", end="2017-09-26").to_pydatetime().tolist()
    + pd.date_range(start="2017-10-06", end="2017-10-06").to_pydatetime().tolist()
    + pd.date_range(start="2017-10-13", end="2017-10-13").to_pydatetime().tolist()
    + pd.date_range(start="2017-11-03", end="2017-11-03").to_pydatetime().tolist()
    + pd.date_range(start="2017-12-21", end="2017-12-31").to_pydatetime().tolist()
    + pd.date_range(start="2018-01-01", end="2018-02-18").to_pydatetime().tolist()
    + pd.date_range(start="2018-02-24", end="2018-02-24").to_pydatetime().tolist()
    + pd.date_range(start="2018-04-30", end="2018-04-30").to_pydatetime().tolist()
    + pd.date_range(start="2018-06-01", end="2018-06-01").to_pydatetime().tolist()
    + pd.date_range(start="2018-07-14", end="2018-07-29").to_pydatetime().tolist()
    + pd.date_range(start="2018-10-01", end="2018-10-01").to_pydatetime().tolist()
    + pd.date_range(start="2018-11-16", end="2018-11-16").to_pydatetime().tolist()
    + pd.date_range(start="2018-12-20", end="2018-12-31").to_pydatetime().tolist()
    + pd.date_range(start="2019-01-01", end="2019-02-13").to_pydatetime().tolist()
    + pd.date_range(start="2019-03-04", end="2019-03-04").to_pydatetime().tolist()
    + pd.date_range(start="2019-03-06", end="2019-03-06").to_pydatetime().tolist()
    + pd.date_range(start="2019-07-13", end="2019-07-28").to_pydatetime().tolist()
    + pd.date_range(start="2019-10-05", end="2019-10-05").to_pydatetime().tolist()
    + pd.date_range(start="2019-12-20", end="2019-12-31").to_pydatetime().tolist()
)
