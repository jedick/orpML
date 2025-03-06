from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from .transform import *

# Dummy regressor model
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

# Histogram-based gradient boosting regression tree
knnreg = Pipeline(
    preprocessor.steps + 
    [("regressor", KNeighborsRegressor(n_neighbors = 3))]
)

