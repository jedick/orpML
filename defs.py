import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
import json

# Function to list the n most abundant taxa
def ListTopTaxa(X, n):

    """
    List the n most abundant taxa at each rank
    
    Parameters:
    X (pd.DataFrame): Input dataframe with abundances and Zc columns
    n (int): Number of most abundant taxa to keep
    
    Returns:
    Column names to keep (both abundances and Zc)
    """
    # Extract the abundances for all ranks
    all_abundances = X[X.columns[X.columns.str.contains("__abundance")]]
    top_cols = []
    for rank in ["domain", "phylum", "class", "order", "family", "genus"]:
        # Extract the abundances for this rank
        abundances = all_abundances[all_abundances.columns[all_abundances.columns.str.contains(rank + "__")]]
        # Calculate relative abundance of taxa in each sample
        relabund = abundances.div(abundances.sum(axis=1), axis=0)
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

class KeepTopTaxa(BaseEstimator, TransformerMixin):

    """Keep columns with the n most abundant taxa.
    -----
    attributes
    fit: identify columns - in the training set
    transform: only use those columns
    """

    def __init__(self, n):
        self.n = n
        self.columns_to_keep = None

    def fit(self, X, y=None):
        self.columns_to_keep = ListTopTaxa(X, self.n)
        return self

    def transform(self, X, y=None, **kwargs):
        return X[self.columns_to_keep]

    def get_params(self, deep=False):
        return {"n": self.n}

def PrintOnce(message = None):

    """Prints a message only once during GridSearchCV.
    Use message = None to reset the saved messages file.
    """

    msgfile = "/tmp/global_messages.json"

    if message == None:
        # Initialize the messages with an empty list
        messages = []
    else:
        try:
            # Load messages from the file
            with open(msgfile, 'r') as f:
                messages = json.load(f)
        except:
            # If the file is unreadable, set messages to empty list
            messages = []

    if not message == None:
        # Print the message if it's not listed in the file
        if not any(message in x for x in messages):
            messages.append(message)
            print(message)

    # Save all the used messages to the file
    with open(msgfile, 'w') as f:
        # indent=2 is not needed but makes the file human-readable if the data is nested
        json.dump(messages, f, indent=2)


# Function to select feature columns at a specified rank
# 'abundance' and 'Zc' should be None or a taxonomic rank
# Or Zc = 'same' to use same rank as abundance
def SelectFeatureCols(X, abundance = None, Zc = None):
    abundance_cols = []
    Zc_cols = []
    if Zc == 'same':
        Zc = abundance
    if not abundance == None:
        abundance_cols = X.columns[X.columns.str.contains("__abundance")]
        abundance_cols = abundance_cols[abundance_cols.str.contains(abundance + "__")]
    if not Zc == None:
        Zc_cols = X.columns[X.columns.str.contains("__Zc")]
        Zc_cols = Zc_cols[Zc_cols.str.contains(Zc + "__")]
    # Unpack each list and combine them
    feature_cols = [*abundance_cols, *Zc_cols]
    # Print the number of features and the parameters
    message = str(len(feature_cols)) + " features for abundance = " + str(abundance) + " and Zc = " + str(Zc)
    PrintOnce(message)
    return feature_cols

class SelectFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, abundance = None, Zc = None):
        self.abundance = abundance
        self.Zc = Zc
        self.feature_cols = None

    def fit(self, X, y = None):
        self.feature_cols = SelectFeatureCols(X, self.abundance, self.Zc)
        return self

    def transform(self, X, y = None, **kwargs):
        if self.feature_cols == []:
            # If abundance and Zc are both None, return a column of zeros
            X = pd.DataFrame(np.zeros((X.shape[0], 1)))
        else:
            # Keep specified columns
            X = X[self.feature_cols]
        return X

    def get_params(self, deep = False):
        return {"abundance": self.abundance, "Zc": self.Zc}


# Transformer to drop columns with frequency of NA values above a certain threshold
# (All-NA columns can occur for Zc for rare taxa in train-test split or CV folds)
class DropNACols(BaseEstimator, TransformerMixin):

    def __init__(self, threshold = 1.0):
        self.na_cols = None
        self.threshold = threshold

    def fit(self, X, y = None):
        # Count and get the frequency of NA values in each column
        count_na = X.isna().sum(axis = 0)
        freq_na = count_na / X.shape[0]
        # If the frequency is greater than or equal to the threshold, mark this as a column to drop
        self.na_cols = X.columns[freq_na >= self.threshold].tolist()
        if len(self.na_cols) > 0:
            # Print the number of features to drop
            if self.threshold == 1.0:
                message = "Dropping " + str(len(self.na_cols)) + " features with all NA values"
            else:
                message = "Dropping " + str(len(self.na_cols)) + " features with NA frequency >= " + str(self.threshold)
            PrintOnce(message)
        return self

    def transform(self, X, y = None, **kwargs):
        X = X.drop(self.na_cols, axis = 1)
        return X

    def get_params(self, deep = False):
        return {"threshold": self.threshold}


