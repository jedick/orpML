# orpML

This repository has files for the study:
*Machine learning prediction of redox potential from microbial abundances and genomes*.

The usage is briefly described below.

## Get data (optional)

NOTE: The repo contains the combined data file (`Bacteria.csv.xz`) that is used for downstream processing.
The following steps only need to be run if you want to recreate this file.

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

## Run models with scikit-learn

Running python in the root directory:

```python
from load import *
# Save results of different grid searches
Step_1_regressor()
Step_2_reduce_dim()
Step_3_n_components()
Step_4_features()
Step_5_hyper()
# Save predictions on test set
Step_6_test()
```

## Evaluating and plotting results

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

## Implementation in a data engineering workflow

For better reusability and maintenance, a modular design is adopted that follows a modified extract-transform-load (ETL) workflow.
The regression model is defined after data transformation (preprocessing), so the workflow becomes ETML.
Lastly, we make plots to evaluate the models, making the entire workflow ETMLP (extract-transform-model-load-plot).

### `extract.py`
This module extracts the data from the `data/` directory and creates the following object for downstream processing:

- `X_train, X_test`: Train and test splits for features (abundance and Zc at all taxonomic ranks)
- `y_train, y_test`: Train and test splits for target (Eh7)

### `transform.py`
This module contains the functions and classes used for preprocessing the data:

- `ListTopTaxa()`: Function to list the n most abundant taxa
- `KeepTopTaxa()`: Transformer class to keep columns with the n most abundant taxa
- `PrintOnce()`: Function to print a message only once during `GridSearchCV()`
- `SelectFeatureCols()`: function to select feature columns for a specified taxonomic rank
- `SelectFeatures()`: Transformer class to select features: abundance and/or Zc each for a given rank
- `DropNACols()`: Transformer class to drop columns with frequency of NA values above a certain threshold

### `model.py`
This module implements the regression models.

### `load.py`
This module is the main Python file that recursively depends on the previous ones.
The first 5 functions carry out grid searches over selected parameters and load the results into the `results/` directory.
The last function saves the predictions of the tuned model on the test set to `test_results.csv`.

- `Step_1_regressor()`
- `Step_2_reduce_dim()`
- `Step_3_n_components()`
- `Step_4_features()`: Feature selection
- `Step_5_hyper()`: Hyperparameter tuning
- `Step_6_test()`: Make predictions on test set

### `plot.py`
The functions in this file require the saved results from all the steps in `load.py`.
These results are saved in the repository, so the plots can be made without having to rerun the previous functions.
