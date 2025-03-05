import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib default colors
# https://www.statology.org/matplotlib-default-colors/
blue = '#1f77b4'
orange = '#ff7f0e'
green = '#2ca02c'
purple = '#9467bd'

def Plot_1_regressor():

    df = read_df_1and2("regressor")

    colors = [blue, orange, orange, green, green, purple, purple]
    styles = [":", "--", "-", "--", "-", "--", "-"]
    # Set color and style of lines in subplot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color = colors, ls = styles)

    df.plot(ax = ax)
    labels=['Baseline', 'Lin. Regression (1)', 'Lin. Regression (2)', 'Random Forests (1)', 'Random Forests (2)', 'HistGBR (1)', 'HistGBR (2)']
    ax.legend(labels=labels, loc='upper left')
    plt.ylabel('Mean absolute error of Eh7 (mV)')
    plt.title("Featuresets: Abundance (1), Abundance + Zc (2)")
    plt.show()


def Plot_2_reduce_dim():

    df = read_df_1and2("reduce_dim")

    colors = [blue, orange, orange, green, green, purple, purple]
    styles = [":", "--", "-", "--", "-", "--", "-"]
    # Set color and style of lines in subplot
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color = colors, ls = styles)

    df.plot(ax = ax)
    labels=['Baseline', 'Lin. Regression (1)', 'Lin. Regression (2)', 'Random Forests (1)', 'Random Forests (2)', 'HistGBR (1)', 'HistGBR (2)']
    ax.legend(labels=labels, loc='upper left')
    plt.ylabel('Mean absolute error of Eh7 (mV)')
    plt.title("Featuresets: Abundance (1), Abundance + Zc (2)")
    plt.show()


def read_df_1and2(directory):

    # Read files with grid search results
    data = {
        'rank': pd.read_csv("results/"+directory+"/dumreg_1.csv").param_selectfeatures__abundance,
        'dumreg_1': pd.read_csv("results/"+directory+"/dumreg_1.csv").mean_test_score,
        'linreg_1': pd.read_csv("results/"+directory+"/linreg_1.csv").mean_test_score,
        'linreg_2': pd.read_csv("results/"+directory+"/linreg_2.csv").mean_test_score,
        'ranfor_1': pd.read_csv("results/"+directory+"/ranfor_1.csv").mean_test_score,
        'ranfor_2': pd.read_csv("results/"+directory+"/ranfor_2.csv").mean_test_score,
        'histgbr_1': pd.read_csv("results/"+directory+"/histgbr_1.csv").mean_test_score,
        'histgbr_2': pd.read_csv("results/"+directory+"/histgbr_2.csv").mean_test_score,
    }

    # Create data frame
    df = pd.concat(data, axis = 1)
    df = df.set_index('rank')
    # Take opposite to plot mean absolute error
    df = -df
    # Set outrageous values to NA
    df[df > 1000] = np.nan
    return df

