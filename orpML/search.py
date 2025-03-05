import ast
from sklearn.metrics import mean_absolute_error
from .extract import *
from .model import *

def run_grid_search(estimator, fileprefix, directory, param_grid, cv = 5, verbose = 1):

    # Function to perform and save grid searches over multiple parameter grids
    PrintOnce()
    print("--- Running GridSearchCV() for", fileprefix, "---")
    grid = GridSearchCV(estimator = estimator, param_grid = param_grid, cv = cv, verbose = verbose, scoring = 'neg_mean_absolute_error', n_jobs = 7)
    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)
    results.to_csv(directory + "/" + fileprefix + ".csv", index = False)

def search_1_no_Zc():

    # Setup grid search for one feature set (abundance)
    preprocessor.set_params(selectfeatures__Zc = None)
    param_grid = {
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
    }

    run_grid_search(dumreg, "dumreg", "results/no_Zc", param_grid)
    run_grid_search(linreg, "linreg", "results/no_Zc", param_grid)
    run_grid_search(ranfor, "ranfor", "results/no_Zc", param_grid)
    run_grid_search(histgbr, "histgbr", "results/no_Zc", param_grid)

def search_2_with_Zc():

    # Setup grid search for two feature sets (abundance + Zc at the same rank)
    preprocessor.set_params(selectfeatures__Zc = "same")
    param_grid = {
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
    }

    run_grid_search(dumreg, "dumreg", "results/with_Zc", param_grid)
    run_grid_search(linreg, "linreg", "results/with_Zc", param_grid)
    run_grid_search(ranfor, "ranfor", "results/with_Zc", param_grid)
    run_grid_search(histgbr, "histgbr", "results/with_Zc", param_grid)

def search_3_abundance_vs_Zc():

    # Two feature sets (abundance + Zc at different ranks)
    param_grid = {
        'selectfeatures__abundance': [None, "phylum", "class", "order", "family", "genus"],
        # Don't include genus for Zc because there are no differences between samples at this rank
        'selectfeatures__Zc': [None, "phylum", "class", "order", "family"],
    }

    run_grid_search(histgbr, "histgbr", "results/abundance_vs_Zc", param_grid)

def search_4_hyperparameters():

    # Get best-performing ranks for abundance and Zc from previous step
    abundance = pd.read_csv("results/abundance_vs_Zc/histgbr.csv").sort_values('rank_test_score')['param_selectfeatures__abundance'].iloc[0]
    preprocessor.set_params(selectfeatures__abundance = abundance)
    Zc = pd.read_csv("results/abundance_vs_Zc/histgbr.csv").sort_values('rank_test_score')['param_selectfeatures__Zc'].iloc[0]
    preprocessor.set_params(selectfeatures__Zc = Zc)
    # Setup hyperparameter search
    param_grid = {
        'regressor__max_iter': [100, 200, 400],
        'regressor__l2_regularization': [0, 0.01, 0.1],
    }

    run_grid_search(histgbr, "histgbr", "results/hyperparameters", param_grid, verbose = 2)

def search_5_test_predictions():

    # Test tuned model
    df = pd.read_csv("results/hyperparameters/histgbr.csv")
    params_text = df.sort_values('rank_test_score')['params'].iloc[0]
    # Convert text to dictionary
    params = ast.literal_eval(params_text)
    histgbr.set_params(**params)
    histgbr.fit(X_train, y_train)

    # Make predictions
    y_train_pred = histgbr.predict(X_train)
    y_test_pred = histgbr.predict(X_test)
    # Evaluate predictions using mean absolute error of Eh7
    MAE_train = mean_absolute_error(y_train, y_train_pred)
    MAE_test = mean_absolute_error(y_test, y_test_pred)
    description = "histgbr"
    print(f"MAE of {description} model on training set: {MAE_train:.3f}")
    print(f"MAE of {description} model on test set: {MAE_test:.3f}")

    # Combine metadata and test and predicted values of Eh7
    y_pred = histgbr.predict(X_test)
    y_pred_df = pd.DataFrame({'Eh7_pred': y_pred})
    y_test_df = pd.DataFrame(y_test)
    # Reset indices before doing pd.concat
    metadata_test.reset_index(drop = True, inplace = True)
    y_test_df.reset_index(drop = True, inplace = True)
    df = pd.concat([metadata_test, y_test_df, y_pred_df], axis = 1)
    df.to_csv("results/test_results.csv", index = False)

