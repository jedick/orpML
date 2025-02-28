import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

# Function to make predictions on test set and print and return MAE
def test_it(model, description):
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Evaluate predictions using mean absolute error of Eh7
    MAE_train = mean_absolute_error(y_train, y_train_pred)
    MAE_test = mean_absolute_error(y_test, y_test_pred)
    print(f"R2 of {description} model on training set: {MAE_train:.3f}")
    print(f"R2 of {description} model on test set: {MAE_test:.3f}")
    return MAE_train, MAE_test

