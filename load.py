import ast
from sklearn.metrics import mean_absolute_error
from model import *

def Step_1_regressor():

    # Setup grid search for one feature set (abundance)
    param_grid_1 = {
        'keeptoptaxa__n': [150],
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
        'selectfeatures__Zc': [None],
        'reduce_dim': ['passthrough'],
    }
    # Setup grid search for two feature sets (abundance + Zc at the same rank)
    param_grid_2 = {
        'keeptoptaxa__n': [150],
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
        'selectfeatures__Zc': ["same"],
        'reduce_dim': ['passthrough'],
    }
    param_grids = [param_grid_1, param_grid_2]

    run_grid_search(dumreg, "dumreg", "results/regressor", param_grids)
    run_grid_search(linreg, "linreg", "results/regressor", param_grids)
    run_grid_search(ranfor, "ranfor", "results/regressor", param_grids)
    run_grid_search(histgbr, "histgbr", "results/regressor", param_grids)

def Step_2_reduce_dim():

    # Setup grid search for one feature set (abundance)
    param_grid_1 = {
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
        'selectfeatures__Zc': [None],
    }
    # Setup grid search for two feature sets (abundance + Zc at the same rank)
    param_grid_2 = {
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
        'selectfeatures__Zc': ["same"],
    }
    param_grids = [param_grid_1, param_grid_2]

    run_grid_search(dumreg, "dumreg", "results/reduce_dim", param_grids)
    run_grid_search(linreg, "linreg", "results/reduce_dim", param_grids)
    run_grid_search(ranfor, "ranfor", "results/reduce_dim", param_grids)
    run_grid_search(histgbr, "histgbr", "results/reduce_dim", param_grids)

def Step_3_n_components():

    # Setup grid search for number of components
    # Use family because it is best performing rank in Step_2
    param_grid_1 = {
        'selectfeatures__abundance': ["family"],
        'selectfeatures__Zc': [None],
        'reduce_dim__n_components': np.arange(50, 255, 5),
    }
    param_grid_2 = {
        'selectfeatures__abundance': ["family"],
        'selectfeatures__Zc': ["family"],
        'reduce_dim__n_components': np.arange(50, 255, 5),
    }
    param_grids = [param_grid_1, param_grid_2]
    run_grid_search(histgbr, "histgbr", "results/n_components", param_grids, verbose = 2)

def Step_4_features():

    # One feature set (abundance)
    # NOTUSED: Use best-performing number of components from previous step
    # Instead, use 150 for both to isolate effects of other processing parameters
    param_grid_1 = {
        #'reduce_dim__n_components':
        #    [pd.read_csv("results/n_components/histgbr_1.csv").sort_values('rank_test_score')['param_reduce_dim__n_components'].iloc[0]],
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
        'selectfeatures__Zc': [None],
    }
    # Two feature sets (abundance + Zc at different ranks)
    param_grid_2 = {
        #'reduce_dim__n_components':
        #    [pd.read_csv("results/n_components/histgbr_2.csv").sort_values('rank_test_score')['param_reduce_dim__n_components'].iloc[0]],
        'selectfeatures__abundance': ["phylum", "class", "order", "family", "genus"],
        # Don't include genus for Zc because there are no differences between samples at this rank
        'selectfeatures__Zc': ["phylum", "class", "order", "family"],
    }
    param_grids = [param_grid_1, param_grid_2]
    run_grid_search(histgbr, "histgbr", "results/features", param_grids)

def Step_5_hyper():

    # One feature set
    param_grid_1 = {
        'selectfeatures__abundance':
            [pd.read_csv("results/features/histgbr_1.csv").sort_values('rank_test_score')['param_selectfeatures__abundance'].iloc[0]],
        'selectfeatures__Zc': [None],
        'regressor__max_iter': [100, 200, 400],
        'regressor__l2_regularization': [0, 0.01, 0.1],
    }
    # Two feature sets
    param_grid_2 = {
        'selectfeatures__abundance':
            [pd.read_csv("results/features/histgbr_2.csv").sort_values('rank_test_score')['param_selectfeatures__abundance'].iloc[0]],
        'selectfeatures__Zc':
            [pd.read_csv("results/features/histgbr_2.csv").sort_values('rank_test_score')['param_selectfeatures__Zc'].iloc[0]],
        'regressor__max_iter': [100, 200, 400],
        'regressor__l2_regularization': [0, 0.01, 0.1],
    }
    param_grids = [param_grid_1, param_grid_2]
    run_grid_search(histgbr, "histgbr", "results/hyper", param_grids, verbose = 2)

def Step_6_test():

    # Test tuned model for one feature set
    df = pd.read_csv("results/hyper/histgbr_1.csv")
    params_1_text = df.sort_values('rank_test_score')['params'].iloc[0]
    # Convert text to dictionary
    params_1 = ast.literal_eval(params_1_text)
    histgbr.set_params(**params_1)
    histgbr.fit(X_train, y_train)
    test_it(histgbr, "one feature set")

    # Test tuned model for two feature sets
    df = pd.read_csv("results/hyper/histgbr_2.csv")
    params_2_text = df.sort_values('rank_test_score')['params'].iloc[0]
    params_2 = ast.literal_eval(params_2_text)
    histgbr.set_params(**params_2)
    histgbr.fit(X_train, y_train)
    test_it(histgbr, "two feature sets")

    # Combine metadata and test and predicted values of Eh7
    y_pred = histgbr.predict(X_test)
    y_pred_df = pd.DataFrame({'Eh7_pred': y_pred})
    y_test_df = pd.DataFrame(y_test)
    # Reset indices before doing pd.concat
    metadata_test.reset_index(drop=True, inplace=True)
    y_test_df.reset_index(drop=True, inplace=True)
    df = pd.concat([metadata_test, y_test_df, y_pred_df], axis = 1)
    df.to_csv("results/test_results.csv", index = False)

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

