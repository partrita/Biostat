**다운로드**
============

**Windows**
-----------

`Github Release <https://github.com/hikarimusic/BIOSTATS/releases/latest/>`_에서 최신 버전을 다운로드할 수 있습니다. *BIOSTATS.exe* 파일을 선택하십시오.

.. image:: ../_static/download/windows.png

또는 아래 링크에서 다운로드할 수 있습니다:

* https://github.com/hikarimusic/BIOSTATS/releases/latest/download/BIOSTATS.exe

다운로드한 파일을 더블 클릭하여 BIOSTATS를 실행합니다.

.. note::
    *Windows*가 실행을 차단하는 경우, *추가 정보 > 실행*을 누르십시오.
    BIOSTATS에는 바이러스가 전혀 없으며, `Github Repository <https://github.com/hikarimusic/BIOSTATS>`_에서 모든 코드를 확인할 수 있습니다.

.. note::
    BIOSTATS를 여는 데 시간이 오래 걸릴 수 있습니다. 이는 일반적으로 컴퓨터의 바이러스 검사 프로세스 때문이며,
    이에 대해 제가 할 수 있는 일은 없습니다. 잠시 기다리면서 차 한잔 하세요!

**Linux**
---------

`Github Release <https://github.com/hikarimusic/BIOSTATS/releases/latest/>`_에서 최신 버전을 다운로드할 수 있습니다. *BIOSTATS* 파일을 선택하십시오.

.. image:: ../_static/download/linux.png

또는 아래 링크에서 다운로드할 수 있습니다:

* https://github.com/hikarimusic/BIOSTATS/releases/latest/download/BIOSTATS

다운로드한 파일을 더블 클릭하여 BIOSTATS를 실행합니다. 또는 BIOSTATS가 있는 디렉토리로 ``cd``한 다음 ``./BIOSTATS``로 프로그램을 실행할 수 있습니다.

.. note::
   Linux에서는 프로그램을 실행하기 전에 ``chmod +x BIOSTATS``로 실행 권한을 허용해야 할 수 있습니다.

**Python 패키지**
------------------

Python에 익숙한 사용자는 Python 패키지를 다운로드하여 대화형 모드 또는 자체 프로젝트에서 사용할 수 있습니다.
패키지를 다운로드하려면 터미널에서 다음 명령을 실행하십시오:

.. code-block:: 

   pip install biostatistics

어떤 디렉토리에서든 ``biostats`` 명령으로 메인 창을 직접 열 수 있습니다:

.. code-block:: 

   biostats

또는 터미널에서 BIOSTATS를 대화형으로 사용할 수 있습니다:

.. code-block::

   :~$ python3
   >>> import biostats as bs
   >>> data = bs.dataset("one_way_anova.csv")
   >>> summary, result = bs.one_way_anova(data=data, variable="Length", between="Location")
   >>> summary
        Location  Count      Mean  Std. Deviation  95% CI: Lower  95% CI: Upper
   1   Tillamook     10  0.080200        0.011963       0.071642       0.088758
   2     Newport      8  0.074800        0.008597       0.067613       0.081987
   3  Petersburg      7  0.103443        0.016209       0.088452       0.118434
   4     Magadan      8  0.078012        0.012945       0.067190       0.088835
   5   Tvarminne      6  0.095700        0.012962       0.082098       0.109302
   >>> result
             D.F.  Sum Square  Mean Square  F Statistic   p-value     
   Location     4    0.004520     0.001130     7.121019  0.000281  ***
   Residual    34    0.005395     0.000159          NaN       NaN  NaN

또는 BIOSTATS를 자체 프로젝트에 통합할 수 있습니다:

.. code-block:: python

    import biostats as bs
    
    data = bs.dataset("one_way_anova.csv")
    summary, result = bs.one_way_anova(data=data, variable="Length", between="Location")
    print(summary)
    print(result)