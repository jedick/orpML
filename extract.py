import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read data
df = pd.read_csv("data/Bacteria.csv.xz")
# Drop rows with missing values for the target variable
target_variable = "Eh7"
df = df[df[target_variable].notna()]
# Get target variable and remaining columns (metadata + features)
y = df[target_variable]
X = df.drop(columns = target_variable)
# Split dataset into training and test sets
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# Separate the metadata from the features
metadata_columns = ['Run', 'Study', 'Name', 'Environment']
metadata_train = X_train[metadata_columns]
metadata_test = X_test[metadata_columns]
X_train.drop(columns = metadata_columns, inplace = True)
X_test.drop(columns = metadata_columns, inplace = True)

