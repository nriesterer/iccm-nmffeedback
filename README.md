ICCM Syllogistic Feedback via JNMF
==================================

Companion repository for the 2020 article "Feedback Influences Syllogistic Strategy: An Analysis based on Joint Nonnegative Matrix Factorization" published in the proceedings of the 18th International Conference on Cognitive Modeling.

### Overview

- `analysis/data/`: Contains the dataset.
- `analysis/fit_results/`: Contains the fit results (JNMF W and H matrices).
- `analysis/results/`: Contains the generated image files.
- `analysis/batchprocjnmf.py`: JNMF implementation.
- `analysis/gen_patterns.py`: Routine for generating the JNMF W and H matrices.
- `analysis/gen_tbl1_data.py`: Generates the data for table 1 (group reassignment).
- `analysis/gen_tbl2_data.py`: Generaets the data for table 2 (correctness).
- `analysis/Makefile`: Makefile to facilitate script execution (`patterns` and `plot_patterns`).
- `analysis/plot_fig1.py`: Generates figure 1 (W matrix patterns).
- `analysis/plot_fig2.py`: Generates figure 2 (pattern model performances).

### Dependencies

- Python 3
    - [CCOBRA](https://github.com/CognitiveComputationLab/ccobra)
    - [matplotlib](https://matplotlib.org)
    - [numpy](https://numpy.org)
    - [pandas](https://pandas.pydata.org)
    - [seaborn](https://seaborn.pydata.org)

### Quickstart

All data/figure generation scripts require W and H matrices to be contained in the `analysis/fit_results` folder. The matrices used to create the paper data are included as defaults. Steps for regenerating these matrices are detailed below.

#### Generate the JNMF results (W and H matrices)

Navigate to the analysis folder and execute the `gen_patterns.py` script:

```
$> cd /path/to/repository/analysis
$> python3 gen_patterns.py data/ccobra_control.csv data/ccobra_exp3_1s_full.csv
$> python3 gen_patterns.py data/ccobra_control.csv data/ccobra_exp3_10s.csv
$> python3 gen_patterns.py data/ccobra_exp3_1s_full.csv data/ccobra_exp3_10s.csv
```

#### Generate the Figures

To generate the images for figure 1, execute the `plot_fig1.py` script for all dataset-combinations:

```
$> cd /path/to/repository/analysis
$> python3 plot_fig1.py ccobra_control ccobra_exp3_1s_full
$> python3 plot_fig1.py ccobra_control ccobra_exp3_10s
$> python3 plot_fig1.py ccobra_exp3_1s_full ccobra_exp3_10s
```

To generate figure 2, execute the `plot_fig2.py` script:

```
$> cd /path/to/repository/analysis
$> python3 plot_fig2.py
```

### Generate the Table Data

To generate the table data, call the respective data generation script:

```
$> cd /path/to/repository/analysis
$> python3 gen_tbl1_data.py
$> python3 gen_tbl2_data.py
```

### Reference

Riesterer, N., Brand, D., & Ragni, M. (in press). Feedback Influences Syllogistic Strategy: An Analysis based on Joint Nonnegative Matrix Factorization. In Proceedings of the 18th International Conference on Cognitive Modeling.
