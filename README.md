# orpML

This repository has files for the study:
*Machine learning prediction of redox potential from microbial abundances and genomes*.

The name `orpML` comes from the acronyms for oxidation-reduction potential (ORP) and machine learning.
- The goal of this study is to develop methods for predicting ORP from DNA sequence data for microbial communities.
- Reliable predictions of ORP would contribute to understanding microbial responses to natural or anthropogenic environmental shifts.
- This information about environments-microbial coupling could result in better management of microbial communities for agricultural or biotechnological purposes.

The features (X values) in the dataset consist of microbial abundances augmented with genome-derived features:
- Relative abundances of microbial groups at different taxonomic ranks were inferred from 16S rRNA gene sequences.
- Carbon oxidation state of proteins at the same taxnomic ranks were computed from reference microbial genomes, weighted by microbial abundances.

The targets (y values) in the dataset are measured ORP values in millivolts (mV) reported in various studies.
The source of data is the study by [Dick and Meng (2023)](https://doi.org/10.1128/msystems.00014-23).

The usage is briefly described below.

## Interactive use

This imports the package and sets up the data preprocessor to use abundances of phyla as the features.
Then we fit a HistGBR (Histogram-Based Gradient Boosting Regressor) model to the data, make predictions, and calculate the mean absolute error using the test set.

```python
from orpML import *
from sklearn.metrics import mean_absolute_error
preprocessor.set_params(selectfeatures__abundance = "phylum")
histgbr.fit(X_train, y_train)
y_pred = histgbr.predict(X_test)
print(mean_absolute_error(y_pred, y_test))
```

The result shows that the predictions of ORP are accurate to within ca. 75 mV on average.

## Running the scripted workflows

Workflows have been implemented to evaluate the performance of different preprocessing steps and model hyperparameters.

### Run models with scikit-learn

```python
# Save results of different grid searches
Step_1_regressor()
Step_2_reduce_dim()
Step_3_n_components()
Step_4_features()
Step_5_hyper()
# Save predictions on test set
Step_6_test()
```

### Evaluating and plotting results

```python
from plot import *
# MAE of Eh7 vs taxonomic rank for different models
# (baseline, linear regression, random forests, HistGBR)
# and feature sets (only abundance or abundance+Zc)
Plot_1_regressor()
# Same as above, with dimension reduction step added
Plot_2_reduce_dim()
```

Run this R script from the root directory of the package:

```R
source("plot.R")
# Test score as a function of number of components using HistGBR
plot_n_components()
# Scatterplot of predicted vs ground-truth Eh7
plot_test()
```

### Get data (optional)

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

## Modular design

For better reusability and maintenance, a modular design is adopted that follows a modified extract-transform-load (ETL) workflow.
The regression models (both traditional and deep-learning) take the output data transformation (preprocessing), so the workflow becomes ETML.
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
This module implements the traditional regression models made with scikit-learn.

### `deep_model.py`
This module implements the deep learning models made with PyTorch.

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
