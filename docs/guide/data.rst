**데이터 처리**
===============


데이터 열기
-------------

You can open the data by pressing *Open* button in *Data* window:

.. image:: ../_static/guide/open_data_1.png
   :width: 500

.. image:: ../_static/guide/open_data_2.png
   :width: 500

File types that can be opened by BIOSTATS:

+------------+------------+------------------------------------------------------------------------------------------------------+
| File Type  | Extension  |Sample Data                                                                                           |
+============+============+======================================================================================================+
| Excel File | .xlsx      |`sample.xlsx <https://github.com/hikarimusic/BIOSTATS/raw/main/examples/sample/sample.xlsx>`_         |
+------------+------------+------------------------------------------------------------------------------------------------------+
| CSV File   | .csv       |`sample.csv <https://github.com/hikarimusic/BIOSTATS/raw/main/examples/sample/sample.csv>`_           |
+------------+------------+------------------------------------------------------------------------------------------------------+
| JSON File  | .json      |`sample.json <https://github.com/hikarimusic/BIOSTATS/raw/main/examples/sample/sample.json>`_         |
+------------+------------+------------------------------------------------------------------------------------------------------+
| SAS File   | .sas7bdat  |`sample.sas7bat <https://github.com/hikarimusic/BIOSTATS/raw/main/examples/sample/sample.sas7bat>`_   |
+------------+------------+------------------------------------------------------------------------------------------------------+
| Stata File | .dta       |`sample.dta <https://github.com/hikarimusic/BIOSTATS/raw/main/examples/sample/sample.dta>`_           |
+------------+------------+------------------------------------------------------------------------------------------------------+
| SPSS File  | .sav       |`sample.sav <https://github.com/hikarimusic/BIOSTATS/raw/main/examples/sample/sample.sav>`_           |
+------------+------------+------------------------------------------------------------------------------------------------------+

.. tip::

    ``o`` 키를 눌러 데이터를 열 수 있습니다.

데이터 편집
-------------

You can edit the data by pressing *Edit* button in *Data* window. and confirm the change by pressing *Confirm* button:

.. image:: ../_static/guide/edit_data_1.png
   :width: 500

.. image:: ../_static/guide/edit_data_2.png
   :width: 500

You can change the number of rows and columns by adjusting the spin boxes above, and change the width of cells by adjusting the scale bar below:

.. image:: ../_static/guide/edit_data_3.png
   :width: 500

.. image:: ../_static/guide/edit_data_4.png
   :width: 500

.. tip::

    You can press ``e`` to edit the data. In the edit mode, you can use the arrow keys ``↑ ↓ ← →`` to move to neighboring cells, and press the enter key ``↵`` to confirm the change.

.. warning::

    편집 모드의 최대 셀 수는 성능 문제로 인해 300개로 제한됩니다. BIOSTATS는 데이터 편집용으로 설계되지 않았으므로 이 목적을 위해서는 *Excel* 또는 *Google Sheets*와 같은 다른 소프트웨어를 사용해야 합니다.

데이터 저장
-------------

You can save the data by pressing *Save* button in *Data* window:

.. image:: ../_static/guide/save_data_1.png
   :width: 500

.. image:: ../_static/guide/save_data_2.png
   :width: 500

File types that can be saved by BIOSTATS:

+----------------+------------+
| File Type      | Extension  |
+================+============+
| Excel File     | .xlsx      |
+----------------+------------+
| CSV File       | .csv       |
+----------------+------------+
| JSON File      | .json      |
+----------------+------------+
| Stata File     | .dta       |
+----------------+------------+
| LaTex File     | .tex       |
+----------------+------------+
| Markdown FIle  | .md        |
+----------------+------------+
| Text File      | .txt       |
+----------------+------------+

.. tip::

    You can press ``Ctrl + s`` to save the data.