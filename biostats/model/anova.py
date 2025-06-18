import pandas as pd
from scipy import stats as st

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA

from biostats.model.util import _CC, _process, _add_p

def one_way_anova(data, variable, between):
    '''
    여러 그룹 간에 변수의 평균값이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 수치형 열과 하나 이상의 범주형 열을 포함해야 합니다.
    variable : :py:class:`str`
        평균값을 계산하려는 수치형 변수입니다.
    between : :py:class:`str`
        표본이 속한 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹의 개수, 평균값, 표준 편차 및 신뢰 구간입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 자유도, 제곱합, 제곱 평균, F 통계량 및 p-값입니다.

    참고 항목
    --------
    one_way_ancova : 다른 변수가 통제될 때 그룹 간 평균값이 다른지 검정합니다.
    two_way_anova : 두 가지 방식으로 분류될 때 그룹 간 평균값이 다른지 검정합니다.
    kruskal_wallis_test : 일원 분산 분석의 비모수적 버전입니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("one_way_anova.csv")
    >>> data
        Length    Location
    0   0.0571   Tillamook
    1   0.0813   Tillamook
    2   0.0831   Tillamook
    3   0.0976   Tillamook
    4   0.0817   Tillamook
    5   0.0859   Tillamook
    6   0.0735   Tillamook
    7   0.0659   Tillamook
    8   0.0923   Tillamook
    9   0.0836   Tillamook
    10  0.0873     Newport
    11  0.0662     Newport
    12  0.0672     Newport
    13  0.0819     Newport
    14  0.0749     Newport
    15  0.0649     Newport
    16  0.0835     Newport
    17  0.0725     Newport
    18  0.0974  Petersburg
    19  0.1352  Petersburg
    20  0.0817  Petersburg
    21  0.1016  Petersburg
    22  0.0968  Petersburg
    23  0.1064  Petersburg
    24  0.1050  Petersburg
    25  0.1033     Magadan
    26  0.0915     Magadan
    27  0.0781     Magadan
    28  0.0685     Magadan
    29  0.0677     Magadan
    30  0.0697     Magadan
    31  0.0764     Magadan
    32  0.0689     Magadan
    33  0.0703   Tvarminne
    34  0.1026   Tvarminne
    35  0.0956   Tvarminne
    36  0.0973   Tvarminne
    37  0.1039   Tvarminne
    38  0.1045   Tvarminne

    We want to test whether the mean values of *Length* in each *Location* are different.

    >>> summary, result = bs.one_way_anova(data=data, variable="Length", between="Location")
    >>> summary
         Location  Count      Mean  Std. Deviation  95% CI: Lower  95% CI: Upper
    1   Tillamook     10  0.080200        0.011963       0.071642       0.088758
    2     Newport      8  0.074800        0.008597       0.067613       0.081987
    3  Petersburg      7  0.103443        0.016209       0.088452       0.118434
    4     Magadan      8  0.078012        0.012945       0.067190       0.088835
    5   Tvarminne      6  0.095700        0.012962       0.082098       0.109302

    The mean values of *Length* and their 95% confidence intervals in each group are given.

    >>> result
              D.F.  Sum Square  Mean Square  F Statistic   p-value     
    Location     4    0.004520     0.001130     7.121019  0.000281  ***
    Residual    34    0.005395     0.000159          NaN       NaN  NaN

    p-값이 0.001보다 작으므로 각 그룹의 *Length* 평균값은 유의하게 다릅니다.

    '''

    data = data[list({variable, between})].dropna()
    _process(data, num=[variable], cat=[between])

    if str(data[variable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(variable))
    if data[between].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between))

    group = data[between].dropna().unique()

    summary = pd.DataFrame()

    for x in group:
        n = _CC(lambda: data[data[between]==x][variable].dropna().count())
        mean = _CC(lambda: data[data[between]==x][variable].dropna().mean())
        std = _CC(lambda: data[data[between]==x][variable].dropna().std())
        sem = _CC(lambda: data[data[between]==x][variable].dropna().sem())
        temp = pd.DataFrame(
            {
                "{}".format(between): _CC(lambda: x),
                "Count": _CC(lambda: n),
                "Mean": _CC(lambda: mean),
                "Std. Deviation": _CC(lambda: std),
                "95% CI: Lower" : _CC(lambda: st.t.ppf(0.025, n-1, mean, sem)) ,
                "95% CI: Upper" : _CC(lambda: st.t.ppf(0.975, n-1, mean, sem)) ,
            }, index=[0]
        )
        summary = pd.concat([summary, temp], ignore_index=True)
    summary.index += 1

    formula = "Q('%s') ~ " % variable
    formula += "C(Q('%s'))" % between
    model = ols(formula, data=data).fit()
    result = anova_lm(model)
    result = result.rename(columns={
        'df': 'D.F.',
        'sum_sq' : 'Sum Square',
        'mean_sq' : 'Mean Square',
        'F' : 'F Statistic',
        'PR(>F)' : 'p-value'
    })
    result = result.rename(index={
        "C(Q('%s'))" % between : between
    })
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result


def two_way_anova(data, variable, between_1, between_2):
    '''
    그룹이 두 가지 방식으로 분류될 때 여러 그룹 간에 변수의 평균값이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 수치형 열과 두 개의 범주형 열을 포함해야 합니다.
    variable : :py:class:`str`
        평균값을 계산하려는 수치형 변수입니다.
    between_1 : :py:class:`str`
        표본의 그룹을 지정하는 첫 번째 범주형 변수입니다. 최대 20개 그룹입니다.
    between_2 : :py:class:`str`
        표본의 그룹을 지정하는 두 번째 범주형 변수입니다. 최대 20개 그룹입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹 조합의 개수, 평균값, 표준 편차 및 신뢰 구간입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 자유도, 제곱합, 제곱 평균, F 통계량 및 p-값입니다.

    참고 항목
    --------
    two_way_ancova : 다른 변수가 통제되는 이원 분산 분석입니다.
    one_way_anova : 그룹 간 평균값이 다른지 검정합니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("two_way_anova.csv")
    >>> data
        Activity     Sex Genotype
    0      1.884    male       ff
    1      2.283    male       ff
    2      2.396    male       fs
    3      2.838  female       ff
    4      2.956    male       fs
    5      4.216  female       ff
    6      3.620  female       ss
    7      2.889  female       ff
    8      3.550  female       fs
    9      3.105    male       fs
    10     4.556  female       fs
    11     3.087  female       fs
    12     4.939    male       ff
    13     3.486    male       ff
    14     3.079  female       ss
    15     2.649    male       fs
    16     1.943  female       fs
    17     4.198  female       ff
    18     2.473  female       ff
    19     2.033  female       ff
    20     2.200  female       fs
    21     2.157  female       fs
    22     2.801    male       ss
    23     3.421    male       ss
    24     1.811  female       ff
    25     4.281  female       fs
    26     4.772  female       fs
    27     3.586  female       ss
    28     3.944  female       ff
    29     2.669  female       ss
    30     3.050  female       ss
    31     4.275    male       ss
    32     2.963  female       ss
    33     3.236  female       ss
    34     3.673  female       ss
    35     3.110    male       ss

    We want to test that whether the mean values of *Activity* are different between *male* and *female*, and between *ff*, *fs* and *ss*. We also want to test that whether there is an interaction between *Sex* and *Genotype*.

    >>> summary, result = bs.two_way_anova(data=data, variable="Activity", between_1="Sex", between_2="Genotype")
    >>> summary
          Sex Genotype  Count     Mean  Std. Deviation  95% CI: Lower  95% CI: Upper
    1    male       ff      4  3.14800        1.374512       0.960845       5.335155
    2    male       fs      4  2.77650        0.316843       2.272332       3.280668
    3    male       ss      4  3.40175        0.634811       2.391624       4.411876
    4  female       ff      8  3.05025        0.959903       2.247751       3.852749
    5  female       fs      8  3.31825        1.144539       2.361392       4.275108
    6  female       ss      8  3.23450        0.361775       2.932048       3.536952

    The mean values and 95% confidence intervals of each combination of *Sex* and *Genotype* are given.

    >>> result
                    D.F.  Sum Square  Mean Square  F Statistic   p-value      
    Sex                1    0.068080     0.068080     0.086128  0.771180  <NA>
    Genotype           2    0.277240     0.138620     0.175366  0.840004  <NA>
    Sex : Genotype     2    0.814641     0.407321     0.515295  0.602515  <NA>
    Residual          30   23.713823     0.790461          NaN       NaN  <NA>

    *Sex*의 p-값이 0.05보다 크므로 두 *Sex* 간에 *Activity*는 다르지 않습니다. *Genotype*의 p-값이 0.05보다 크므로 세 *Genotype* 간에 *Activity*는 다르지 않습니다. 상호 작용의 p-값이 0.05보다 크므로 *Sex*와 *Genotype* 간에 상호 작용이 없습니다.

    '''

    data = data[list({variable, between_1, between_2})].dropna()
    _process(data, num=[variable], cat=[between_1, between_2])

    if str(data[variable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(variable))
    if data[between_1].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between_1))
    if data[between_2].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between_2))

    group_1 = data[between_1].dropna().unique()
    group_2 = data[between_2].dropna().unique()

    summary = pd.DataFrame()

    for x in group_1:
        for y in group_2:
            n = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][variable].dropna().count())
            mean = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][variable].dropna().mean())
            std = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][variable].dropna().std())
            sem = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][variable].dropna().sem())
            temp = pd.DataFrame(
                {
                    "{}".format(between_1): _CC(lambda: x),
                    "{}".format(between_2): _CC(lambda: y),
                    "Count": _CC(lambda: n),
                    "Mean": _CC(lambda: mean),
                    "Std. Deviation": _CC(lambda: std),
                    "95% CI: Lower" : _CC(lambda: st.t.ppf(0.025, n-1, mean, sem)) ,
                    "95% CI: Upper" : _CC(lambda: st.t.ppf(0.975, n-1, mean, sem)) ,
                }, index=[0]
            )
            summary = pd.concat([summary, temp], ignore_index=True)
    summary.index += 1

    formula = "Q('%s') ~ " % variable
    formula += "C(Q('%s'), Sum) * " % between_1
    formula += "C(Q('%s'), Sum)" % between_2
    model = ols(formula, data=data).fit()
    result = anova_lm(model)
    result = result.rename(columns={
        'df': 'D.F.',
        'sum_sq' : 'Sum Square',
        'mean_sq' : 'Mean Square',
        'F' : 'F Statistic',
        'PR(>F)' : 'p-value'
    })
    index_change = {}
    for index in result.index:
        changed = index
        changed = changed.replace("C(Q('%s'), Sum)" % between_1, between_1)
        changed = changed.replace("C(Q('%s'), Sum)" % between_2, between_2)
        changed = changed.replace(":", " : ")
        index_change[index] = changed
    result = result.rename(index_change)
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result


def one_way_ancova(data, variable, between, covariable):
    '''
    다른 변수가 통제될 때 여러 그룹 간에 변수의 평균값이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 수치형 열과 하나의 범주형 열을 포함해야 합니다.
    variable : :py:class:`str`
        평균값을 계산하려는 수치형 변수입니다.
    between : :py:class:`str`
        표본이 속한 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다.
    covariable : :py:class:`str`
        통제하려는 다른 수치형 변수입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹에서 변수의 개수, 평균값 및 표준 편차와 공변량의 개수, 평균값 및 표준 편차입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 제곱합, 자유도, F 통계량 및 p-값입니다.

    참고 항목
    --------
    one_way_anova : 그룹 간 평균값이 다른지 검정합니다.
    two_way_ancova : 다른 변수가 통제될 때 두 가지 방식으로 분류된 그룹 간 평균값이 다른지 검정합니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("one_way_ancova.csv")
    >>> data
        Pulse Species  Temp
    0    67.9      ex  20.8
    1    65.1      ex  20.8
    2    77.3      ex  24.0
    3    78.7      ex  24.0
    4    79.4      ex  24.0
    5    80.4      ex  24.0
    6    85.8      ex  26.2
    7    86.6      ex  26.2
    8    87.5      ex  26.2
    9    89.1      ex  26.2
    10   98.6      ex  28.4
    11  100.8      ex  29.0
    12   99.3      ex  30.4
    13  101.7      ex  30.4
    14   44.3     niv  17.2
    15   47.2     niv  18.3
    16   47.6     niv  18.3
    17   49.6     niv  18.3
    18   50.3     niv  18.9
    19   51.8     niv  18.9
    20   60.0     niv  20.4
    21   58.5     niv  21.0
    22   58.9     niv  21.0
    23   60.7     niv  22.1
    24   69.8     niv  23.5
    25   70.9     niv  24.2
    26   76.2     niv  25.9
    27   76.1     niv  26.5
    28   77.0     niv  26.5
    29   77.7     niv  26.5
    30   84.7     niv  28.6
    31   74.3    fake  17.2
    32   77.2    fake  18.3
    33   77.6    fake  18.3
    34   79.6    fake  18.3
    35   80.3    fake  18.9
    36   81.8    fake  18.9
    37   90.0    fake  20.4
    38   88.5    fake  21.0
    39   88.9    fake  21.0
    40   90.7    fake  22.1
    41   99.8    fake  23.5
    42  100.9    fake  24.2
    43  106.2    fake  25.9
    44  106.1    fake  26.5
    45  107.0    fake  26.5
    46  107.7    fake  26.5
    47  114.7    fake  28.6

    We want to test whether the mean values of *Pulse* ar different between the three *Species*, with *Temp* being controlled.

    >>> summary, result = bs.one_way_ancova(data=data, variable="Pulse", between="Species", covariable="Temp")
    >>> summary
      Species  Count  Mean (Pulse)  Std. (Pulse)  Mean (Temp)  Std. (Temp)
    1      ex     14     85.585714      11.69930    25.757143     3.074639
    2     niv     17     62.429412      12.95684    22.123529     3.659325
    3    fake     17     92.429412      12.95684    22.123529     3.659325

    The mean values of *Pulse* and *Temp* in each group are given.

    >>> result
               Sum Square  D.F.  F Statistic       p-value     
    Species   7835.737962     2  1372.995165  2.252680e-40  ***
    Temp      7025.952857     1  2462.205692  2.877499e-40  ***
    Residual   125.554874    44          NaN           NaN  NaN

    *Species*의 p-값이 0.05보다 작으므로 *Temp*가 통제된 후에도 세 *Species* 간에 *Pulse*의 평균값은 다릅니다.

    '''

    data = data[list({variable, between, covariable})].dropna()
    _process(data, num=[variable, covariable], cat=[between])

    if str(data[variable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(variable))
    if data[between].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between))
    if str(data[covariable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(covariable))

    group = data[between].dropna().unique()

    summary = pd.DataFrame()

    for x in group:
        n = _CC(lambda: data[data[between]==x][[variable, covariable]].dropna()[variable].count())
        mean_1 = _CC(lambda: data[data[between]==x][[variable, covariable]].dropna()[variable].mean())
        std_1 = _CC(lambda: data[data[between]==x][[variable, covariable]].dropna()[variable].std())
        mean_2 = _CC(lambda: data[data[between]==x][[variable, covariable]].dropna()[covariable].mean())
        std_2 = _CC(lambda: data[data[between]==x][[variable, covariable]].dropna()[covariable].std())
        temp = pd.DataFrame(
            {
                "{}".format(between): x,
                "Count": n,
                "Mean ({})".format(variable): _CC(lambda: mean_1),
                "Std. ({})".format(variable): _CC(lambda: std_1),
                "Mean ({})".format(covariable): _CC(lambda: mean_2),
                "Std. ({})".format(covariable): _CC(lambda: std_2),
            }, index=[0]
        )
        summary = pd.concat([summary, temp], ignore_index=True)
    summary.index += 1

    formula = "Q('%s') ~ " % variable
    formula += "C(Q('%s'), Sum) + " % between
    formula += "Q('%s')" % covariable
    model = ols(formula, data=data).fit()

    result = anova_lm(model, typ=2)
    result = result.rename(columns={
        'df': 'D.F.',
        'sum_sq' : 'Sum Square',
        'F' : 'F Statistic',
        'PR(>F)' : 'p-value'
    })
    index_change = {}
    for index in result.index:
        changed = index
        changed = changed.replace("C(Q('%s'), Sum)" % between, between)
        changed = changed.replace("Q('%s')" % covariable, covariable)
        index_change[index] = changed
    result = result.rename(index_change)
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result


def two_way_ancova(data, variable, between_1, between_2, covariable):
    '''
    다른 변수가 통제될 때 두 가지 방식으로 분류된 여러 그룹 간에 변수의 평균값이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 수치형 열과 두 개의 범주형 열을 포함해야 합니다.
    variable : :py:class:`str`
        평균값을 계산하려는 수치형 변수입니다.
    between_1 : :py:class:`str`
        표본의 그룹을 지정하는 첫 번째 범주형 변수입니다. 최대 20개 그룹입니다.
    between_2 : :py:class:`str`
        표본의 그룹을 지정하는 두 번째 범주형 변수입니다. 최대 20개 그룹입니다.
    covariable : :py:class:`str`
        통제하려는 다른 수치형 변수입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹 조합에서 변수의 개수, 평균값 및 표준 편차와 공변량의 개수, 평균값 및 표준 편차입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 제곱합, 자유도, F 통계량 및 p-값입니다.

    참고 항목
    --------
    two_way_anova : 두 가지 방식으로 분류된 여러 그룹 간 평균값이 다른지 검정합니다.
    one_way_ancova : 다른 변수가 통제될 때 그룹 간 평균값이 다른지 검정합니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("two_way_ancova.csv")
    >>> data
        Activity     Sex Genotype  Age
    0      1.884    male       ff   69
    1      2.283    male       ff   51
    2      2.396    male       fs   75
    3      2.838  female       ff   68
    4      2.956    male       fs   29
    5      4.216  female       ff   28
    6      3.620  female       ss   56
    7      2.889  female       ff   38
    8      3.550  female       fs   32
    9      3.105    male       fs   61
    10     4.556  female       fs   20
    11     3.087  female       fs   57
    12     4.939    male       ff   71
    13     3.486    male       ff   21
    14     3.079  female       ss   43
    15     2.649    male       fs   62
    16     1.943  female       fs   54
    17     4.198  female       ff   45
    18     2.473  female       ff   27
    19     2.033  female       ff   66
    20     2.200  female       fs   74
    21     2.157  female       fs   19
    22     2.801    male       ss   20
    23     3.421    male       ss   75
    24     1.811  female       ff   68
    25     4.281  female       fs   25
    26     4.772  female       fs   38
    27     3.586  female       ss   18
    28     3.944  female       ff   49
    29     2.669  female       ss   18
    30     3.050  female       ss   34
    31     4.275    male       ss   49
    32     2.963  female       ss   42
    33     3.236  female       ss   25
    34     3.673  female       ss   55
    35     3.110    male       ss   73

    We want to test that whether the mean values of Activity are different between male and female, and between ff, fs and ss, with *Age* being controlled.

    >>> summary, result = bs.two_way_ancova(data=data, variable="Activity", between_1="Sex", between_2="Genotype", covariable="Age")
    >>> summary
          Sex Genotype  Count  Mean (Activity)  Std. (Activity)  Mean (Age)  Std. (Age)
    1    male       ff      4          3.14800         1.374512      53.000   23.151674
    2    male       fs      4          2.77650         0.316843      56.750   19.568257
    3    male       ss      4          3.40175         0.634811      54.250   25.708300
    4  female       ff      8          3.05025         0.959903      48.625   17.204132
    5  female       fs      8          3.31825         1.144539      39.875   19.910066
    6  female       ss      8          3.23450         0.361775      36.375   15.202796

    The mean values of *Activity* and *Age* in each combination of groups are given.

    >>> result
                    Sum Square  D.F.  F Statistic   p-value      
    Sex               0.018057     1     0.023349  0.879612  <NA>
    Genotype          0.113591     2     0.073441  0.929363  <NA>
    Sex : Genotype    0.727884     2     0.470606  0.629311  <NA>
    Age               1.286714     1     1.663822  0.207280  <NA>
    Residual         22.427109    29          NaN       NaN  <NA>

    *Age*를 통제한 후 *Sex*의 p-값이 0.05보다 크므로 두 *Sex* 간에 *Activity*는 다르지 않습니다. *Genotype*의 p-값이 0.05보다 크므로 세 *Genotype* 간에 *Activity*는 다르지 않습니다. 상호 작용의 p-값이 0.05보다 크므로 *Sex*와 *Genotype* 간에 상호 작용이 없습니다.

    '''

    data = data[list({variable, between_1, between_2, covariable})].dropna()
    _process(data, num=[variable, covariable], cat=[between_1, between_2])

    if str(data[variable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(variable))
    if data[between_1].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between_1))
    if data[between_2].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between_2))
    if str(data[covariable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(covariable))

    group_1 = data[between_1].dropna().unique()
    group_2 = data[between_2].dropna().unique()

    summary = pd.DataFrame()


    summary = pd.DataFrame()

    for x in group_1:
        for y in group_2:
            n = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][[variable, covariable]].dropna()[variable].count())
            mean_1 = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][[variable, covariable]].dropna()[variable].mean())
            std_1 = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][[variable, covariable]].dropna()[variable].std())
            mean_2 = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][[variable, covariable]].dropna()[covariable].mean())
            std_2 = _CC(lambda: data[(data[between_1]==x) & (data[between_2]==y)][[variable, covariable]].dropna()[covariable].std())
            temp = pd.DataFrame(
                {
                    "{}".format(between_1): _CC(lambda: x),
                    "{}".format(between_2): _CC(lambda: y),
                    "Count": _CC(lambda: n),
                    "Mean ({})".format(variable): _CC(lambda: mean_1),
                    "Std. ({})".format(variable): _CC(lambda: std_1),
                    "Mean ({})".format(covariable): _CC(lambda: mean_2),
                    "Std. ({})".format(covariable): _CC(lambda: std_2),
                }, index=[0]
            )
            summary = pd.concat([summary, temp], ignore_index=True)
    summary.index += 1

    formula = "Q('%s') ~ " % variable
    formula += "C(Q('%s'), Sum) * " % between_1
    formula += "C(Q('%s'), Sum) + " % between_2
    formula += "Q('%s')" % covariable
    model = ols(formula, data=data).fit()

    result = anova_lm(model, typ=2)
    result = result.rename(columns={
        'df': 'D.F.',
        'sum_sq' : 'Sum Square',
        'F' : 'F Statistic',
        'PR(>F)' : 'p-value'
    })
    index_change = {}
    for index in result.index:
        changed = index
        changed = changed.replace("C(Q('%s'), Sum)" % between_1, between_1)
        changed = changed.replace("C(Q('%s'), Sum)" % between_2, between_2)
        changed = changed.replace("Q('%s')" % covariable, covariable)
        changed = changed.replace(":", " : ")
        index_change[index] = changed
    result = result.rename(index_change)
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result

def multivariate_anova(data, variable, between):
    '''
    여러 그룹 간에 여러 변수의 평균값이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 수치형 열과 하나의 범주형 열을 포함해야 합니다.
    variable : :py:class:`list`
        평균값을 계산하려는 수치형 변수의 목록입니다.
    between : :py:class:`str`
        표본이 속한 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹에서 각 수치형 변수의 평균값과 표준 편차입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 자유도, 필라이의 트레이스, F 통계량 및 p-값입니다.

    참고 항목
    --------
    one_way_anova : 그룹 간 변수의 평균값이 다른지 검정합니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("multivariate_anova.csv")
    >>> data
         sepal_length  sepal_width    species
    0             5.1          3.5     setosa
    1             4.9          3.0     setosa
    2             4.7          3.2     setosa
    3             4.6          3.1     setosa
    4             5.0          3.6     setosa
    ..            ...          ...        ...
    145           6.7          3.0  virginica
    146           6.3          2.5  virginica
    147           6.5          3.0  virginica
    148           6.2          3.4  virginica
    149           5.9          3.0  virginica

    We want to test whether the mean values of *sepal_length* and *sepal_width* in each *species* are different.

    >>> summary, result = bs.multivariate_anova(data=data, variable=["sepal_length", "sepal_width"], between="species")
    >>> summary
          species  Mean (sepal_length)  Std. (sepal_length)  Mean (sepal_width)  Std. (sepal_width)
    1      setosa                5.006             0.352490               3.428            0.379064
    2  versicolor                5.936             0.516171               2.770            0.313798
    3   virginica                6.588             0.635880               2.974            0.322497

    The mean values of *sepal_length* and *sepal_width* in each *species* are given.

    >>> result
             D.F.  Pillai's Trace  F Statistic       p-value     
    species     2        0.945314     65.87798  9.902977e-40  ***

    p-값이 0.001보다 작으므로 각 그룹의 *sepal_length* 및 *sepal_width* 평균값은 유의하게 다릅니다.

    '''

    data = data[list(set(variable + [between]))].dropna()
    _process(data, num=variable, cat=[between])

    for var in variable:
        if str(data[var].dtypes) not in ("float64", "Int64"):
            raise Warning("The column '{}' must be numeric".format(var))
    if data[between].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between))

    group = data[between].dropna().unique().tolist()

    summary = pd.DataFrame({between:group})
    for var in variable:
        mean = []
        std = []
        for x in group:
            mean.append(_CC(lambda: data[data[between]==x][var].dropna().mean()))
            std.append(_CC(lambda: data[data[between]==x][var].dropna().std()))
        summary["Mean ({})".format(var)] = mean
        summary["Std. ({})".format(var)] = std  
    summary.index += 1 

    formula = ""
    for var in variable:
        formula += "{} + ".format(var)
    formula = formula[:-3]
    formula += " ~ {}".format(between)
    fit = MANOVA.from_formula(formula, data=data)
    table = pd.DataFrame((fit.mv_test().results[between]['stat']))
    result = pd.DataFrame(
        {
            "D.F." : _CC(lambda: len(group)-1) ,
            "Pillai's Trace" : _CC(lambda: table.iloc[1][0]) ,
            "F Statistic" : _CC(lambda: table.iloc[1][3]) ,
            "p-value" : _CC(lambda: table.iloc[1][4])
        }, index=[between]
    )
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result

def repeated_measures_anova(data, variable, between, subject):
    '''
    반복 측정 데이터에서 여러 그룹 간에 변수의 평균값이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 수치형 열과 하나의 범주형 열, 그리고 피험자를 지정하는 열을 포함해야 합니다.
    variable : :py:class:`str`
        평균값을 계산하려는 수치형 변수입니다.
    between : :py:class:`str`
        표본이 속한 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다.
    subject : :py:class:`str`
        피험자 ID를 지정하는 변수입니다. 동일한 피험자에서 측정된 표본은 동일한 ID를 가져야 합니다. 최대 2000명의 피험자입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹의 개수, 평균값, 표준 편차 및 신뢰 구간입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 자유도, 제곱합, 제곱 평균, F 통계량 및 p-값입니다.

    참고 항목
    --------
    one_way_anova : 그룹 간 변수의 평균값이 다른지 검정합니다.
    friedman_test : 반복 측정 분산 분석의 비모수적 버전입니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("repeated_measures_anova.csv")
    >>> data
        response drug  patient
    0         30    A        1
    1         28    B        1
    2         16    C        1
    3         34    D        1
    4         14    A        2
    5         18    B        2
    6         10    C        2
    7         22    D        2
    8         24    A        3
    9         20    B        3
    10        18    C        3
    11        30    D        3
    12        38    A        4
    13        34    B        4
    14        20    C        4
    15        44    D        4
    16        26    A        5
    17        28    B        5
    18        14    C        5
    19        30    D        5

    We want to test whether the mean values of *response* in each *drug* are different, when the samples are repeatedly measured on the four *patient*.

    >>> summary, result = bs.repeated_measures_anova(data=data, variable="response", between="drug", subject="patient")
    >>> summary
      drug  Count  Mean  Std. Deviation  95% CI: Lower  95% CI: Upper
    1    A      5  26.4        8.763561      15.518602      37.281398
    2    B      5  25.6        6.542171      17.476822      33.723178
    3    C      5  15.6        3.847077      10.823223      20.376777
    4    D      5  32.0        8.000000      22.066688      41.933312

    The mean values of *response* and their 95% confidence intervals in each group are given.

    >>> result
              D.F.  Sum Square  Mean Square  F Statistic  p-value     
    drug         3       698.2   232.733333    24.758865  0.00002  ***
    Residual    12       112.8     9.400000          NaN      NaN  NaN

    p-값이 0.001보다 작으므로 각 그룹의 *response* 평균값은 유의하게 다릅니다.

    '''

    data = data[list({variable, between, subject})].dropna()
    _process(data, num=[variable], cat=[between, subject])

    if str(data[variable].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(variable))
    if data[between].nunique() > 20:
        raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(between))
    if data[subject].nunique() > 2000:
        raise Warning("The nmuber of classes in column '{}' cannot > 2000.".format(subject))

    cross = pd.crosstab(index=data[subject], columns=data[between])
    for col in cross:
        cross = cross.drop(cross[cross[col] != 1].index)
    sub = cross.index.values.tolist()
    data = data[data[subject].isin(sub)]
    
    group = data[between].dropna().unique()

    summary = pd.DataFrame()

    for x in group:
        n = _CC(lambda: data[data[between]==x][variable].dropna().count())
        mean = _CC(lambda: data[data[between]==x][variable].dropna().mean())
        std = _CC(lambda: data[data[between]==x][variable].dropna().std())
        sem = _CC(lambda: data[data[between]==x][variable].dropna().sem())
        temp = pd.DataFrame(
            {
                "{}".format(between): _CC(lambda: x),
                "Count": _CC(lambda: n),
                "Mean": _CC(lambda: mean),
                "Std. Deviation": _CC(lambda: std),
                "95% CI: Lower" : _CC(lambda: st.t.ppf(0.025, n-1, mean, sem)) ,
                "95% CI: Upper" : _CC(lambda: st.t.ppf(0.975, n-1, mean, sem)) ,
            }, index=[0]
        )
        summary = pd.concat([summary, temp], ignore_index=True)
    summary.index += 1

    formula = "Q('%s') ~ " % variable
    formula += "C(Q('%s'))" % between
    model = ols(formula, data=data).fit()
    anova_1 = anova_lm(model)
    formula = "Q('%s') ~ " % variable
    formula += "C(Q('%s'))" % subject
    model = ols(formula, data=data).fit()
    anova_2 = anova_lm(model)

    df_1 = _CC(lambda: anova_1.iloc[0][0])
    df_2 = _CC(lambda: df_1 * anova_2.iloc[0][0])
    SS_1 = _CC(lambda: anova_1.iloc[0][1])
    SS_2 = _CC(lambda: anova_2.iloc[1][1] - SS_1)
    MS_1 = _CC(lambda: SS_1 / df_1)
    MS_2 = _CC(lambda: SS_2 / df_2)
    F = _CC(lambda: MS_1 / MS_2)
    p = _CC(lambda: 1 - st.f.cdf(F, df_1, df_2))

    result = pd.DataFrame(
        {
            "D.F." : [df_1, df_2] ,
            "Sum Square" : [SS_1, SS_2] ,
            "Mean Square" : [MS_1, MS_2] ,
            "F Statistic" : [F, None] ,
            "p-value" : [p, None]
        }, index = [between, "Residual"]
    )
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result

