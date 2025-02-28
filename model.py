from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from extract import *
from transform import *

# Using PCA before predictive model:
# https://stats.stackexchange.com/questions/258938/pca-before-random-forest-regression-provide-better-predictive-scores-for-my-data

# Define the preprocessing pipeline
preprocessor = Pipeline([
    # nb. Linear regression error explodes with all 500 taxa
    ("keeptoptaxa", KeepTopTaxa(500)),
    ("selectfeatures", SelectFeatures()),
    ("dropnacols", DropNACols()),
    # Imputer converts DataFrame to NumPy array
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("reduce_dim", PCA(150)),
])

# Dummy regressor (baseline) model
dumreg = Pipeline(
    preprocessor.steps + 
    [("regressor", DummyRegressor())]
)

# Linear regression model
linreg = Pipeline(
    preprocessor.steps + 
    [("regressor", LinearRegression())]
)

# Random forest regressor model
ranfor = Pipeline(
    preprocessor.steps + 
    # With n_jobs > 1 in GridSearchCV, we need to random_state here for reproducibility (not np.random.seed above)
    # Use n_estimators = 100 to be consistent with HistGradientBoostingRegressor(max_iter = 100)
    [("regressor", RandomForestRegressor(n_estimators = 100, random_state = 1))]
)

# Histogram-based gradient boosting regression tree
histgbr = Pipeline(
    preprocessor.steps + 
    # Use max_leaf_nodes = None to be consistent with RandomForestRegressor()
    [("regressor", HistGradientBoostingRegressor(max_leaf_nodes = None, random_state = 1))]
)

# Function to perform and save grid searches over multiple parameter grids
def run_grid_search(estimator, fileprefix, directory, param_grids, cv = 5, verbose = 1):
    for i in range(len(param_grids)):
        PrintOnce()
        print("--- Running GridSearchCV() for", fileprefix, "with param_grid", i+1, "---")
        grid = GridSearchCV(estimator = estimator, param_grid = param_grids[i], cv = cv, verbose = verbose, scoring = 'neg_mean_absolute_error', n_jobs = 7)
        grid.fit(X_train, y_train)
        results = pd.DataFrame(grid.cv_results_)
        results.to_csv(directory + "/" + fileprefix + "_" + str(i+1) + ".csv", index = False)

