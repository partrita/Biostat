import matplotlib.pyplot as plt
import seaborn as sns

from biostats.model.util import _CC, _process, _add_p

def histogram(data, x, band, color=None):
    '''
    수치형 변수의 분포를 보여주는 히스토그램을 그립니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 수치형 열을 포함해야 합니다.
    x : :py:class:`str`
        플롯할 수치형 변수입니다.
    band : :py:class:`int`
        히스토그램의 막대 수입니다.
    color : :py:class:`str`
        다른 색상으로 플롯할 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다. 선택 사항입니다.

    반환값
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        생성된 플롯입니다.

    참고 항목
    --------
    density_plot : 밀도 곡선으로 분포를 보여줍니다.
    histogram_2D : 2개의 수치형 열에서 2차원 히스토그램을 그립니다.

    예제
    --------
    .. plot::

        >>> import biostats as bs
        >>> import matplotlib.pyplot as plt
        >>> data = bs.dataset("penguins.csv")
        >>> data
            species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
        0    Adelie  Torgersen            39.1           18.7                181         3750    MALE
        1    Adelie  Torgersen            39.5           17.4                186         3800  FEMALE
        2    Adelie  Torgersen            40.3           18.0                195         3250  FEMALE
        3    Adelie  Torgersen             NaN            NaN               <NA>         <NA>     NaN
        4    Adelie  Torgersen            36.7           19.3                193         3450  FEMALE
        ..      ...        ...             ...            ...                ...          ...     ...
        339  Gentoo     Biscoe             NaN            NaN               <NA>         <NA>     NaN
        340  Gentoo     Biscoe            46.8           14.3                215         4850  FEMALE
        341  Gentoo     Biscoe            50.4           15.7                222         5750    MALE
        342  Gentoo     Biscoe            45.2           14.8                212         5200  FEMALE
        343  Gentoo     Biscoe            49.9           16.1                213         5400    MALE

        We want to visualize the distribution of *flipper_length_mm* in different *species*.

        >>> fig = bs.histogram(data=data, x="flipper_length_mm", band=10, color="species")
        >>> plt.show()

    '''

    sns.set_theme()
    data = data.dropna(how='all')
    _process(data, num=[x], cat=[color])

    if str(data[x].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(x))
    if color:
        if data[color].nunique() > 20:
            raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(color))

    fig, ax = plt.subplots()
    sns.histplot(data=data, x=x, bins=band, hue=color, ax=ax)
    
    return fig


def density_plot(data, x, smooth, color=None):
    '''
    수치형 변수의 분포를 보여주는 밀도 곡선을 그립니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 수치형 열을 포함해야 합니다.
    x : :py:class:`str`
        플롯할 수치형 변수입니다.
    smooth : :py:class:`float` 또는 :py:class:`int`
        곡선의 평활도입니다. 값이 클수록 곡선이 더 부드러워집니다.
    color : :py:class:`str`
        다른 색상으로 플롯할 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다. 선택 사항입니다.

    반환값
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        생성된 플롯입니다.

    참고 항목
    --------
    histogram : 히스토그램으로 분포를 보여줍니다.
    density_plot_2D : 2개의 수치형 열에서 2차원 밀도 플롯을 그립니다.

    예제
    --------
    .. plot::

        >>> import biostats as bs
        >>> import matplotlib.pyplot as plt
        >>> data = bs.dataset("penguins.csv")
        >>> data
            species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
        0    Adelie  Torgersen            39.1           18.7                181         3750    MALE
        1    Adelie  Torgersen            39.5           17.4                186         3800  FEMALE
        2    Adelie  Torgersen            40.3           18.0                195         3250  FEMALE
        3    Adelie  Torgersen             NaN            NaN               <NA>         <NA>     NaN
        4    Adelie  Torgersen            36.7           19.3                193         3450  FEMALE
        ..      ...        ...             ...            ...                ...          ...     ...
        339  Gentoo     Biscoe             NaN            NaN               <NA>         <NA>     NaN
        340  Gentoo     Biscoe            46.8           14.3                215         4850  FEMALE
        341  Gentoo     Biscoe            50.4           15.7                222         5750    MALE
        342  Gentoo     Biscoe            45.2           14.8                212         5200  FEMALE
        343  Gentoo     Biscoe            49.9           16.1                213         5400    MALE

        We want to visualize the distribution of *flipper_length_mm* in different *species*.

        >>> fig = bs.density_plot(data=data, x="flipper_length_mm", smooth=1, color="species")
        >>> plt.show()

    '''

    sns.set_theme()
    data = data.dropna(how='all')
    _process(data, num=[x], cat=[color])

    if str(data[x].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(x))
    if color:
        if data[color].nunique() > 20:
            raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(color))

    fig, ax = plt.subplots()
    sns.kdeplot(data= data, x=x, bw_adjust=smooth, hue=color, ax=ax)
    
    return fig

def cumulative_plot(data, x, color=None):
    '''
    수치형 변수의 분포를 보여주는 누적 곡선을 그립니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 하나 이상의 수치형 열을 포함해야 합니다.
    x : :py:class:`str`
        플롯할 수치형 변수입니다.
    color : :py:class:`str`
        다른 색상으로 플롯할 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다. 선택 사항입니다.

    반환값
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        생성된 플롯입니다.

    참고 항목
    --------
    density_plot: 밀도 곡선으로 분포를 보여줍니다.

    예제
    --------
    .. plot::

        >>> import biostats as bs
        >>> import matplotlib.pyplot as plt
        >>> data = bs.dataset("penguins.csv")
        >>> data
            species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
        0    Adelie  Torgersen            39.1           18.7                181         3750    MALE
        1    Adelie  Torgersen            39.5           17.4                186         3800  FEMALE
        2    Adelie  Torgersen            40.3           18.0                195         3250  FEMALE
        3    Adelie  Torgersen             NaN            NaN               <NA>         <NA>     NaN
        4    Adelie  Torgersen            36.7           19.3                193         3450  FEMALE
        ..      ...        ...             ...            ...                ...          ...     ...
        339  Gentoo     Biscoe             NaN            NaN               <NA>         <NA>     NaN
        340  Gentoo     Biscoe            46.8           14.3                215         4850  FEMALE
        341  Gentoo     Biscoe            50.4           15.7                222         5750    MALE
        342  Gentoo     Biscoe            45.2           14.8                212         5200  FEMALE
        343  Gentoo     Biscoe            49.9           16.1                213         5400    MALE

        We want to visualize the cumulative distribution of *flipper_length_mm* in different *species*.

        >>> fig = bs.density_plot(data=data, x="flipper_length_mm", smooth=1, color="species")
        >>> plt.show()

    '''

    sns.set_theme()
    data = data.dropna(how='all')
    _process(data, num=[x], cat=[color])

    if str(data[x].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(x))
    if color:
        if data[color].nunique() > 20:
            raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(color))

    fig, ax = plt.subplots()
    sns.ecdfplot(data= data, x=x, hue=color, ax=ax)
    
    return fig

def histogram_2D(data, x, y, color=None):
    '''
    두 수치형 변수의 분포를 보여주는 2D 히스토그램을 그립니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 수치형 열을 포함해야 합니다.
    x : :py:class:`str`
        x축에 플롯할 수치형 변수입니다.
    y : :py:class:`str`
        y축에 플롯할 수치형 변수입니다.
    color : :py:class:`str`
        다른 색상으로 플롯할 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다. 선택 사항입니다.

    반환값
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        생성된 플롯입니다.

    참고 항목
    --------
    density_plot_2D : 밀도 곡선으로 2D 분포를 보여줍니다.
    histogram : 수치형 변수의 분포를 보여주는 히스토그램을 그립니다.

    예제
    --------
    .. plot::

        >>> import biostats as bs
        >>> import matplotlib.pyplot as plt
        >>> data = bs.dataset("penguins.csv")
        >>> data
            species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
        0    Adelie  Torgersen            39.1           18.7                181         3750    MALE
        1    Adelie  Torgersen            39.5           17.4                186         3800  FEMALE
        2    Adelie  Torgersen            40.3           18.0                195         3250  FEMALE
        3    Adelie  Torgersen             NaN            NaN               <NA>         <NA>     NaN
        4    Adelie  Torgersen            36.7           19.3                193         3450  FEMALE
        ..      ...        ...             ...            ...                ...          ...     ...
        339  Gentoo     Biscoe             NaN            NaN               <NA>         <NA>     NaN
        340  Gentoo     Biscoe            46.8           14.3                215         4850  FEMALE
        341  Gentoo     Biscoe            50.4           15.7                222         5750    MALE
        342  Gentoo     Biscoe            45.2           14.8                212         5200  FEMALE
        343  Gentoo     Biscoe            49.9           16.1                213         5400    MALE

        We want to visualize the 2D distribution of *bill_depth_mm* and *body_mass_g* in different *species*.

        >>> fig = bs.histogram_2D(data=data, x="bill_depth_mm", y="body_mass_g", color="species")
        >>> plt.show()

    '''

    sns.set_theme()
    data = data.dropna(how='all')
    _process(data, num=[x, y], cat=[color])

    if str(data[x].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(x))
    if str(data[y].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(y))
    if color:
        if data[color].nunique() > 20:
            raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(color))

    fig, ax = plt.subplots()
    sns.histplot(data=data, x=x, y=y, hue=color, ax=ax)
        
    return fig

def density_plot_2D(data, x, y, color=None):
    '''
    두 수치형 변수의 분포를 보여주는 2D 밀도 플롯을 그립니다.

    매개변수
    ----------
    data : :py:class:`pandas.DataFrame`
        입력 데이터입니다. 두 개 이상의 수치형 열을 포함해야 합니다.
    x : :py:class:`str`
        x축에 플롯할 수치형 변수입니다.
    y : :py:class:`str`
        y축에 플롯할 수치형 변수입니다.
    color : :py:class:`str`
        다른 색상으로 플롯할 그룹을 지정하는 범주형 변수입니다. 최대 20개 그룹입니다. 선택 사항입니다.

    반환값
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        생성된 플롯입니다.

    참고 항목
    --------
    histogram_2D : 히스토그램으로 2D 분포를 보여줍니다.
    density_plot : 수치형 변수의 분포를 보여주는 밀도 곡선을 그립니다.

    예제
    --------
    .. plot::

        >>> import biostats as bs
        >>> import matplotlib.pyplot as plt
        >>> data = bs.dataset("penguins.csv")
        >>> data
            species     island  bill_length_mm  bill_depth_mm  flipper_length_mm  body_mass_g     sex
        0    Adelie  Torgersen            39.1           18.7                181         3750    MALE
        1    Adelie  Torgersen            39.5           17.4                186         3800  FEMALE
        2    Adelie  Torgersen            40.3           18.0                195         3250  FEMALE
        3    Adelie  Torgersen             NaN            NaN               <NA>         <NA>     NaN
        4    Adelie  Torgersen            36.7           19.3                193         3450  FEMALE
        ..      ...        ...             ...            ...                ...          ...     ...
        339  Gentoo     Biscoe             NaN            NaN               <NA>         <NA>     NaN
        340  Gentoo     Biscoe            46.8           14.3                215         4850  FEMALE
        341  Gentoo     Biscoe            50.4           15.7                222         5750    MALE
        342  Gentoo     Biscoe            45.2           14.8                212         5200  FEMALE
        343  Gentoo     Biscoe            49.9           16.1                213         5400    MALE

        We want to visualize the 2D distribution of *bill_depth_mm* and *body_mass_g* in different *species*.

        >>> fig = bs.density_plot_2D(data=data, x="bill_depth_mm", y="body_mass_g", color="species")
        >>> plt.show()

    '''

    sns.set_theme()
    data = data.dropna(how='all')
    _process(data, num=[x, y], cat=[color])
    
    if str(data[x].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(x))
    if str(data[y].dtypes) not in ("float64", "Int64"):
        raise Warning("The column '{}' must be numeric".format(y))
    if color:
        if data[color].nunique() > 20:
            raise Warning("The nmuber of classes in column '{}' cannot > 20.".format(color))
            
    fig, ax = plt.subplots()
    sns.kdeplot(data=data, x=x, y=y, hue=color, ax=ax)
        
    return fig
