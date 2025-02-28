# orpML

This repo has files for the study:
*Machine learning prediction of redox potential from microbial abundances and genomes*.

The usage is briefly described below.

## Get data

Running R in the `data/` directory:

```R
source("get_data.R")
get_data(feature = "abundance")
get_data(feature = "Zc")
```

This produces two files: `Bacteria_phylum_abundance.csv` and `Bacteria_phylum_Zc.csv`.
The first five columns of each file are metadata, including the target variable (Eh7).
The remaining columns are either relative abundance values or Zc; the latter is missing if abundance = 0.

```R
combine_files()
```

This combines the previously created .csv files into one file.
Only this file is kept in the repository, as `Bacteria.csv.xz`.

## Fit models with scikit-learn

Running python in the root directory:

```python
from main import *
Step_1_regressor()
Step_2_reduce_dim()
Step_3_n_components()
# Feature selection
Step_4_features()
# Hyperparameter tuning
Step_5_hyper()
# Predictions on test set
Step_6_test()
```

## Plot results

```python
from plot import *
# MAE of Eh7 vs taxonomic rank for different models
# (baseline, linear regression, random forests, HistGBR)
# and feature sets (only abundance or abundance+Zc)
Plot_1_regressor()
# Same as above, with dimension reduction step added
Plot_2_reduce_dim()
```

Running R in the root directory:

```R
source("plot.R")
# Test score as a function of number of components using HistGBR
plot_n_components()
# Scatterplot of predicted vs ground-truth Eh7
plot_test()
```

## Description of Python modules

### `data.py`
- `X_train, X_test`: Train and test splits for features (abundance and Zc at all taxonomic ranks)
- `y_train, y_test`: Train and test splits for target (Eh7)
- `test_it()`: Function to make predictions on test set and print and return MAE

### `defs.py`
- `ListTopTaxa()`: Function to list the n most abundant taxa
- `KeepTopTaxa()`: Transformer class to keep columns with the n most abundant taxa
- `PrintOnce()`: Function to print a message only once during `GridSearchCV()`
- `SelectFeatureCols()`: function to select feature columns for a specified taxonomic rank
- `SelectFeatures()`: Transformer class to select features: abundance and/or Zc each for a given rank
- `DropNACols()`: Transformer class to drop columns with frequency of NA values above a certain threshold


