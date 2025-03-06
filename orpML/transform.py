import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from .util import *

def ListTopTaxa(X, n, rank = "all"):

    """
    List the n most abundant taxa at each
    rank (default) or in a single named rank.
    
    Parameters:
    X (pd.DataFrame): Input dataframe with abundances and Zc columns
    n (int): Number of most abundant taxa to keep
    rank (str): one named rank or "all" for all ranks
    
    Returns:
    Column names to keep (both abundances and Zc)
    """

    # Extract all columns with abundance values
    all_abundances = X[X.columns[X.columns.str.contains("__abundance")]]
    # Start with empty list of columns
    top_cols = []
    # Use all ranks or a single rank
    ranks = ["domain", "phylum", "class", "order", "family", "genus"]
    if not rank == "all":
        ranks = [rank]
    # Loop over ranks
    for rank in ranks:
        # Extract the abundances for this rank
        abundances = all_abundances[all_abundances.columns[all_abundances.columns.str.contains(rank + "__")]]
        # Calculate relative abundance of taxa in each sample
        relabund = abundances.div(abundances.sum(axis = 1), axis = 0)
        # Get mean relative abundance in all samples
        meanabund = relabund.mean(axis = 0)
        # Sort the relative abundances
        sortabund = meanabund.sort_values(ascending = False)
        # Get top n taxa abundance columns
        top_abundance = sortabund.index[:n].tolist()
        # Get Zc columns for the same taxa
        top_Zc = [txt.replace("abundance", "Zc") for txt in top_abundance]
        top_cols = top_cols + top_abundance + top_Zc

    # Return columns with top n taxa
    return top_cols

# https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline
# https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65

class GetTaxa(BaseEstimator, TransformerMixin):

    """
    Keep columns with the n most abundant taxa
    in each rank (default) or in a single rank
    (if 'rank' is not None).
    -----
    attributes
    fit: identify columns - in the training set
    transform: only use those columns
    """

    def __init__(self, n, rank = "all"):
        self.n = n
        self.rank = rank
        self.columns_to_keep = None

    def fit(self, X, y = None):
        self.columns_to_keep = ListTopTaxa(X, self.n, self.rank)
        return self

    def transform(self, X, y = None, **kwargs):
        return X[self.columns_to_keep]

    def get_params(self, deep = False):
        return {"n": self.n, "rank": self.rank}


def ListFeatureCols(X, abundance = None, Zc = None):

    """
    Function to list specific feature columns

    'abundance' and 'Zc' parameters can have the following values:
        None or NaN: list no columns for this feature
        "all": list all columns for this feature
        other string: list columns whose names include this taxonomic rank
    """

    abundance_cols = []
    Zc_cols = []
    # NOTE: pd.notna() is False for both None and NaN (nan) values
    if pd.notna(abundance):
        abundance_cols = X.columns[X.columns.str.contains("__abundance")]
        if not abundance == "all":
            abundance_cols = abundance_cols[abundance_cols.str.contains(abundance + "__")]
    if pd.notna(Zc):
        Zc_cols = X.columns[X.columns.str.contains("__Zc")]
        if not Zc == "all":
            Zc_cols = Zc_cols[Zc_cols.str.contains(Zc + "__")]
    # Unpack each list and combine them
    feature_cols = [*abundance_cols, *Zc_cols]
    # Print the number of features and the parameters
    message = str(len(feature_cols)) + " features for abundance = " + str(abundance) + " and Zc = " + str(Zc)
    PrintOnce(message)
    return feature_cols

class UseAbundance(BaseEstimator, TransformerMixin):

    def __init__(self, rank = None):
        self.rank = rank
        self.feature_cols = None

    def fit(self, X, y = None):
        self.feature_cols = ListFeatureCols(X, abundance = self.rank, Zc = None)
        return self

    def transform(self, X, y = None, **kwargs):
        if self.feature_cols == []:
            # If there are no features, return a column of zeros
            X = pd.DataFrame(np.zeros((X.shape[0], 1)))
        else:
            # Keep specified columns
            X = X[self.feature_cols]
        return X

    def get_params(self, deep = False):
        return {"rank": self.rank}

class UseZc(BaseEstimator, TransformerMixin):

    def __init__(self, rank = None):
        self.rank = rank
        self.feature_cols = None

    def fit(self, X, y = None):
        self.feature_cols = ListFeatureCols(X, abundance = None, Zc = self.rank)
        return self

    def transform(self, X, y = None, **kwargs):
        if self.feature_cols == []:
            # If there are no features, return a column of zeros
            X = pd.DataFrame(np.zeros((X.shape[0], 1)))
        else:
            # Keep specified columns
            X = X[self.feature_cols]
        return X

    def get_params(self, deep = False):
        return {"rank": self.rank}


# Pipeline section to get and process abundance data
abundance_pipe = Pipeline([
    # rank = "all" uses all ranks that are provided by gettaxa in the pipeline
    ("use", UseAbundance(rank = "all")),
    # Use L1 norm to make relative abundances in each sample sum to 1
    ("normalizer", Normalizer(norm = "l1")),
])

# Pipeline section to get and process Zc data
Zc_pipe = Pipeline([
    ("use", UseZc(rank = "all")),
    # Standardize (center) values to 0 mean without scaling
    # NOTE: this converts DataFrame to NumPy array
    ("scaler", StandardScaler()),
    # Replace missing values with zero
    ("imputer", SimpleImputer(strategy = "constant", fill_value = 0)),
])

# Feature union for abundance and Zc data
abundance_Zc_union = FeatureUnion([
    ("abundance", abundance_pipe),
    ("Zc", Zc_pipe),
])

preprocessor = Pipeline([
    # Put the pipeline together, starting with the top 100 taxa in each rank
    ("gettaxa", GetTaxa(n = 100, rank = "all")),
    ("feat", abundance_Zc_union),
])

