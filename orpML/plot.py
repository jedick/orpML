import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib default colors
# https://www.statology.org/matplotlib-default-colors/
blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
purple = "#9467bd"

def plot_1_regressor():

    # Read the results from each search
    df1 = read_results("no_Zc")
    df2 = read_results("with_Zc")
    # Combine (collate) columns from both DataFrames
    df = pd.concat([pd.concat((df1.iloc[:, i], df2.iloc[:, i]), axis = 1) for i in range(4)], axis = 1)
    ## Drop one of the DummyRegressor columns because they"re both the same
    #df = df.iloc[: , 1:]

    colors = [blue, blue, orange, orange, green, green, purple, purple]
    styles = ["--", "-", "--", "-", "--", "-", "--", "-"]
    # Set color and style of lines in subplot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color = colors, ls = styles)

    df.plot(ax = ax)
    labels=["Lin. Reg. (1)", "Lin. Reg. (2)",
            "KNN Reg. (1)", "KNN Reg. (2)",
            "Rand. For. (1)", "Rand. For. (2)",
            "HistGBR (1)", "HistGBR (2)"]
    plt.legend(labels = labels, loc = "upper left", bbox_to_anchor = (1.05, 1.0))
    plt.ylabel("Mean absolute error of Eh7 (mV)")
    plt.title("Featuresets: Abundance (1), Abundance + Zc (2)")
    plt.tight_layout()
    plt.show()


def read_results(directory):

    # Read files with grid search results
    data = {
        "rank": pd.read_csv("results/"+directory+"/dumreg.csv").param_gettaxa__rank,
        "linreg": pd.read_csv("results/"+directory+"/linreg.csv").mean_test_score,
        "knnreg": pd.read_csv("results/"+directory+"/knnreg.csv").mean_test_score,
        "ranfor": pd.read_csv("results/"+directory+"/ranfor.csv").mean_test_score,
        "histgbr": pd.read_csv("results/"+directory+"/histgbr.csv").mean_test_score,
    }

    # Create data frame
    df = pd.concat(data, axis = 1)
    df = df.set_index("rank")
    # Take opposite to plot mean absolute error
    df = -df
    ## Set outrageous values to NA
    #df[df > 1000] = np.nan
    return df

