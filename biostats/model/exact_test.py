import numpy as np
import pandas as pd
import math

from biostats.model.util import _CC, _process, _add_p

class binom_exact:

    def __init__(self, table, freqency):
        self.table = table
        self.freq = freqency

    def calc(self):

        self.sum = sum(self.table)

        self.p_part = math.factorial(self.sum)

        self.p_0 = self.multi_nom(self.table)
        self.p = 0
        #self.cnt = 0 # 카운터

        mat = [0] * len(self.table)
        pos = 0

        self.dfs(mat, pos)

        #if self.cnt > 1000000: # 카운터가 1,000,000을 초과하면
        #    return np.NAN # NaN 반환

        return self.p

    def dfs(self, mat, pos):

        #self.cnt += 1 # 카운터 증가
        #if self.cnt > 1000000: # 카운터가 1,000,000을 초과하면
        #    return # 반환

        mat_new = []
        for x in mat:
            mat_new.append(x)

        if pos == -1:
            temp = self.sum - sum(mat_new)
            if temp <0:
                return
            mat_new[len(mat)-1] = temp

            p_1 = self.multi_nom(mat_new)
            if p_1 <= self.p_0 + 0.000000000000000001:
                self.p += p_1
        else:
            max_ = self.sum - sum(mat_new)
            for k in range(max_+1):
                mat_new[pos] = k
                if pos == len(mat)-2:
                    pos_new = -1
                else:
                    pos_new = pos + 1
                self.dfs(mat_new, pos_new)

    def multi_nom(self, table):
        p = self.p_part
        for i in range(len(table)):
            p *= self.freq[i] ** table[i]
        for x in table:
            p /= math.factorial(x)
        return p

def binomial_test(data, variable, expect):
    '''
    범주형 변수의 비율이 예상 비율과 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 범주형 열을 포함해야 합니다. 최대 500개 행입니다.
    variable : :py:class:`str`
        비율을 계산하려는 범주형 변수입니다. 최대 10개 그룹입니다.
    expect : :py:class:`dict`
        각 그룹의 예상 비율입니다. 비율의 합은 자동으로 1로 정규화됩니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        각 그룹의 관찰된 개수 및 비율과 예상 개수 및 비율입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 p-값입니다.

    참고 항목
    --------
    chi_square_test_fit : 이항 검정의 정규 근사 버전입니다.
    fisher_exact_test : 두 범주형 변수 간의 연관성을 검정합니다.

    참고
    -----
    .. warning::
        이항 검정은 가능한 모든 분포를 반복하여 정확한 p-값을 계산하므로 데이터 크기가 매우 클 경우 시간이 많이 걸릴 수 있습니다. 더 큰 데이터의 경우 :py:func:`chi_square_test_fit`를 사용하는 것이 좋습니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("binomial_test.csv")
    >>> data
         Flower
    0    Purple
    1    Purple
    2       Red
    3      Blue
    4     White
    ..      ...
    143  Purple
    144  Purple
    145    Blue
    146     Red
    147    Blue

    We want to test whether the proportion in *Flower* is different from the expected proportions.

    >>> summary, result = bs.binomial_test(data=data, variable="Flower", expect={"Purple":9, "Red":3, "Blue":3, "White":1})
    >>> summary
            Observe  Prop.(Obs.)  Expect  Prop.(Exp.)
    Purple       72     0.486486   83.25       0.5625
    Red          38     0.256757   27.75       0.1875
    Blue         20     0.135135   27.75       0.1875
    White        18     0.121622    9.25       0.0625

    The observed and expected counts and proportions of each group are given.

    >>> result
            p-value    
    Model  0.002255  **

    p-값이 0.01보다 작으므로 관찰된 비율은 예상 비율과 유의하게 다릅니다.

    '''

    data = data[[variable]].dropna()
    _process(data, cat=[variable])

    if data[variable].nunique() > 10:
        raise Warning("The nmuber of classes in column '{}' cannot > 10.".format(variable))
    if len(data) > 500:
        raise Warning("The length of data cannot > 500.")

    cat = data.groupby(variable, sort=False)[variable].groups.keys()
    obs = []
    exp = []
    pro_o = []
    pro_e = []
    exp_sum = sum(list(expect.values()))
    for var in cat:
        obs_val = _CC(lambda: data[variable].value_counts()[var])
        obs.append(_CC(lambda: obs_val))
        pro_o.append(_CC(lambda: obs_val / len(data)))
        exp.append(_CC(lambda: expect[var] * len(data) / exp_sum))
        pro_e.append(_CC(lambda: expect[var] / exp_sum))

    summary = pd.DataFrame(
        {
            "Observe" : _CC(lambda: obs),
            "Prop.(Obs.)" : _CC(lambda: pro_o),
            "Expect"  : _CC(lambda: exp),
            "Prop.(Exp.)" : _CC(lambda: pro_e),
        }, index=cat
    )

    test = binom_exact(obs, pro_e)
    p = _CC(lambda: test.calc())

    result = pd.DataFrame(
        {
            "p-value": _CC(lambda: p)
        }, index=["Model"]
    )

    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result

class fisher_exact:
    
    def __init__(self, table):
        self.table = table

    def calc(self):

        self.row_sum = []
        self.col_sum = []
        self.sum = 0

        for i in range(len(self.table)):
            temp = 0
            for j in range(len(self.table[0])):
                temp += self.table[i][j]
            self.row_sum.append(temp)
        
        for j in range(len(self.table[0])):
            temp = 0
            for i in range(len(self.table)):
                temp += self.table[i][j]
            self.col_sum.append(temp)
        
        for k in self.row_sum:
            self.sum += k
        
        self.p_part = 1
        for x in self.row_sum:
            self.p_part *= math.factorial(x)
        for y in self.col_sum:
            self.p_part *= math.factorial(y)
        self.p_part /= math.factorial(self.sum)

        self.p_0 = self.hyper_geom(self.table)
        self.p = 0
        #self.cnt = 0 # 카운터

        mat = [[0] * len(self.col_sum)] * len(self.row_sum)
        pos = (0, 0)

        self.dfs(mat, pos)

        #if self.cnt > 1000000: # 카운터가 1,000,000을 초과하면
        #    return np.NAN # NaN 반환

        return self.p

    def dfs(self, mat, pos):

        #self.cnt += 1 # 카운터 증가
        #if self.cnt > 1000000: # 카운터가 1,000,000을 초과하면
        #    return # 반환
        
        (xx, yy) = pos
        (rr, cc) = (len(self.row_sum), len(self.col_sum))

        mat_new = []

        for i in range(len(mat)):
            temp = []
            for j in range(len(mat[0])):
                temp.append(mat[i][j])
            mat_new.append(temp)

        if xx == -1 and yy == -1:
            for i in range(rr-1):
                temp = self.row_sum[i]
                for j in range(cc-1):
                    temp -= mat_new[i][j]
                mat_new[i][cc-1] = temp
            for j in range(cc-1):
                temp = self.col_sum[j]
                for i in range(rr-1):
                    temp -= mat_new[i][j]
                mat_new[rr-1][j] = temp
            temp = self.row_sum[rr-1]
            for j in range(cc-1):
                temp -= mat_new[rr-1][j]
            if temp <0:
                return
            mat_new[rr-1][cc-1] = temp
            
            p_1 = self.hyper_geom(mat_new)

            if p_1 <= self.p_0 + 0.000000000000000001:
                self.p += p_1
        else:
            max_1 = self.row_sum[xx]
            max_2 = self.col_sum[yy]
            for j in range(cc):
                max_1 -= mat_new[xx][j]
            for i in range(rr):
                max_2 -= mat_new[i][yy]
            for k in range(min(max_1,max_2)+1):
                mat_new[xx][yy] = k
                if xx == rr-2 and yy == cc-2:
                    pos_new = (-1, -1)
                elif xx == rr-2:
                    pos_new = (0, yy+1)
                else:
                    pos_new = (xx+1, yy)
                self.dfs(mat_new, pos_new)

    def hyper_geom(self, table):
        p = self.p_part
        for i in range(len(table)):
            for j in range(len(table[0])):
                p /= math.factorial(table[i][j])
        return p

def fisher_exact_test(data, variable_1, variable_2, kind="count"):
    '''
    두 범주형 변수 간에 연관성이 있는지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 범주형 열을 포함해야 합니다.
    variable_1 : :py:class:`str`
        첫 번째 범주형 변수입니다. 최대 10개 그룹입니다.
    variable_2 : :py:class:`str`
        두 번째 범주형 변수입니다. 두 변수를 바꿔도 피셔 정확 검정 결과는 변경되지 않습니다. 최대 10개 그룹입니다.
    kind : :py:class:`str`
        분할표를 요약하는 방법입니다.

        * "count" : 발생 빈도를 계산합니다.
        * "vertical" : 각 열의 합계가 1이 되도록 세로로 비율을 계산합니다.
        * "horizontal" : 각 행의 합계가 1이 되도록 가로로 비율을 계산합니다.
        * "overall" : 전체 테이블의 합계가 1이 되도록 전체 비율을 계산합니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        두 범주형 변수의 분할표입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 p-값입니다.

    참고 항목
    --------
    chi_square_test : 피셔 정확 검정의 정규 근사 버전입니다.
    binomial_test : 변수의 관찰된 비율과 예상 비율 간의 차이를 검정합니다.

    참고
    -----
    .. warning::
        피셔 정확 검정은 가능한 모든 분포를 반복하여 정확한 p-값을 계산하므로 데이터 크기가 매우 클 경우 시간이 많이 걸릴 수 있습니다. 더 큰 데이터의 경우 :py:func:`chi_square_test`를 사용하는 것이 좋습니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("fisher_exact_test.csv")
    >>> data
        Frequency     Result
    0     Monthly  Undamaged
    1     Monthly    Damaged
    2     Monthly    Damaged
    3     Monthly    Damaged
    4     Monthly  Undamaged
    ..        ...        ...
    95    Monthly  Undamaged
    96     Weekly  Undamaged
    97    Monthly    Damaged
    98  Quarterly  Undamaged
    99    Monthly  Undamaged

    We want to test whether there is an association between *Frequency* and *Result*.

    >>> summary, result = bs.fisher_exact_test(data=data, variable_1="Frequency", variable_2="Result", kind="horizontal")
    >>> summary
               Damaged  Undamaged
    Daily         0.04       0.96
    Monthly       0.56       0.44
    Quarterly     0.44       0.56
    Weekly        0.20       0.80

    The proportions of *Damaged* in different *Frequency* are given.

    >>> result
            p-value     
    Model  0.000123  ***

    p-값이 0.001보다 작으므로 *Frequency*와 *Result* 간에 유의한 연관성이 있습니다. 즉, 네 가지 *Frequency* 간에 *Damaged*의 비율이 다릅니다.

    '''

    data = data[list({variable_1, variable_2})].dropna()
    _process(data, cat=[variable_1, variable_2])

    if data[variable_1].nunique() > 10:
        raise Warning("The nmuber of classes in column '{}' cannot > 10.".format(variable_1))
    if data[variable_2].nunique() > 10:
        raise Warning("The nmuber of classes in column '{}' cannot > 10.".format(variable_2))

    summary = pd.crosstab(index=data[variable_1], columns=data[variable_2])
    summary.index.name = None
    summary.columns.name = None

    obs = summary.values.tolist()

    if kind == "vertical":
        col_sum = _CC(lambda: summary.sum(axis=0))
        for i in range(summary.shape[0]):
            for j in range(summary.shape[1]):
                summary.iat[i,j] = _CC(lambda: summary.iat[i,j] / col_sum[j])

    if kind == "horizontal":
        col_sum = _CC(lambda: summary.sum(axis=1))
        for i in range(summary.shape[0]):
            for j in range(summary.shape[1]):
                summary.iat[i,j] = _CC(lambda: summary.iat[i,j] / col_sum[i])

    if kind == "overall":
        _sum = _CC(lambda: summary.to_numpy().sum())
        for i in range(summary.shape[0]):
            for j in range(summary.shape[1]):
                summary.iat[i,j] = _CC(lambda: summary.iat[i,j] / _sum)

    test = fisher_exact(obs)
    p = _CC(lambda: test.calc())

    result = pd.DataFrame(
        {
            "p-value": _CC(lambda: p)
        }, index=["Model"]
    )
    
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result

def mcnemar_exact_test(data, variable_1, variable_2, pair):
    '''
    두 쌍을 이룬 그룹에서 범주형 변수의 비율이 다른지 검정합니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 범주형 열과 쌍을 지정하는 열을 포함해야 합니다.
    variable_1 : :py:class:`str`
        표본이 속한 그룹을 지정하는 범주형 변수입니다. 최대 10개 그룹입니다. 가장 빈번하게 나타나는 두 그룹이 자동으로 선택됩니다.
    variable_2 : :py:class:`str`
        비율을 계산하려는 범주형 변수입니다. 최대 10개 그룹입니다. 가장 빈번하게 나타나는 두 그룹이 자동으로 선택됩니다.
    pair : :py:class:`str`
        쌍 ID를 지정하는 변수입니다. 동일한 쌍의 표본은 동일한 ID를 가져야 합니다. 최대 1000쌍입니다.

    반환값
    -------
    summary : :py:class:`pandas.DataFrame`
        일치된 쌍을 단위로 하는 두 범주형 변수의 분할표입니다.
    result : :py:class:`pandas.DataFrame`
        검정의 p-값입니다.

    참고 항목
    --------
    mcnemar_test : 맥니마 검정의 정규 근사 버전입니다.
    fisher_exact_test : 두 범주형 변수 간의 연관성을 검정합니다.

    예제
    --------
    >>> import biostats as bs
    >>> data = bs.dataset("mcnemar_exact_test.csv")
    >>> data
       Treatment Result  ID
    0    control   fail   1
    1    control   fail   2
    2    control   fail   3
    3    control   fail   4
    4    control   fail   5
    ..       ...    ...  ..
    83      test   pass  40
    84      test   pass  41
    85      test   pass  42
    86      test   pass  43
    87      test   pass  44

    We want to test whether the proportions of *Result* are different between the two *Treatment*, where each *control* is paired with a *test*.

    >>> summary, result = bs.mcnemar_exact_test(data=data, variable_1="Treatment", variable_2="Result", pair="ID")
    >>> summary
                    test : fail  test : pass
    control : fail           21            9
    control : pass            2           12

    The contingency table of *Treatment* and *Result* where the counting unit is the matched pair.

    >>> result
           p-value      
    Model  0.06543  <NA>

    p-값이 0.05보다 크므로 두 *Treatment* 하에서 *Result*의 비율 간에 유의한 차이가 없습니다.

    '''

    data = data[list({variable_1, variable_2, pair})].dropna()
    _process(data, cat=[variable_1, variable_2, pair])
    
    if data[variable_1].nunique() > 10:
        raise Warning("The nmuber of classes in column '{}' cannot > 10.".format(variable_1))
    if data[variable_2].nunique() > 10:
        raise Warning("The nmuber of classes in column '{}' cannot > 10.".format(variable_2))
    if data[pair].nunique() > 1000:
        raise Warning("The nmuber of classes in column '{}' cannot > 1000.".format(pair))

    grp_1 = data[variable_1].value_counts()[:2].index.tolist()
    grp_2 = data[variable_2].value_counts()[:2].index.tolist()

    data = data[data[variable_1].isin(grp_1)]
    data = data[data[variable_2].isin(grp_2)]

    cross = pd.crosstab(index=data[pair], columns=data[variable_1])
    for col in cross:
        cross = cross.drop(cross[cross[col] != 1].index)
    sub = cross.index.values.tolist()
    data = data[data[pair].isin(sub)]

    _dat = pd.DataFrame(
        {
            "fst" : data[data[variable_1]==grp_1[0]].sort_values(by=[pair])[variable_2].tolist() ,
            "snd" : data[data[variable_1]==grp_1[1]].sort_values(by=[pair])[variable_2].tolist()
        }
    )

    a = _CC(lambda: _dat[(_dat["fst"]==grp_2[0]) & (_dat["snd"]==grp_2[0])]["fst"].count())
    b = _CC(lambda: _dat[(_dat["fst"]==grp_2[0]) & (_dat["snd"]==grp_2[1])]["fst"].count())
    c = _CC(lambda: _dat[(_dat["fst"]==grp_2[1]) & (_dat["snd"]==grp_2[0])]["fst"].count())
    d = _CC(lambda: _dat[(_dat["fst"]==grp_2[1]) & (_dat["snd"]==grp_2[1])]["fst"].count())

    summary = pd.DataFrame(
        {
            "{} : {}".format(grp_1[1], grp_2[0]) : [a, c] ,
            "{} : {}".format(grp_1[1], grp_2[1]) : [b, d] ,
        }, index=["{} : {}".format(grp_1[0], grp_2[0]), "{} : {}".format(grp_1[0], grp_2[1])]
    )

    p = _CC(lambda: 0)
    n = _CC(lambda: b + c)
    if b < c:
        for x in range(0, b+1):
            p = _CC(lambda: p + math.factorial(n) / (math.factorial(x) * math.factorial(n-x) * 2**n))
        p = _CC(lambda: p * 2)
    elif b > c:
        for x in range(b, n+1):
            p = _CC(lambda: p + math.factorial(n) / (math.factorial(x) * math.factorial(n-x) * 2**n))
        p = _CC(lambda: p * 2)
    else: 
        p = _CC(lambda: 1)

    result = pd.DataFrame(
        {
            "p-value": _CC(lambda: p)
        }, index=["Model"]
    )
    _add_p(result)

    _process(summary)
    _process(result)

    return summary, result