import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing

training_data_path = "house-prices-advanced-regression-techniques/train.csv"
testing_data_path = "house-prices-advanced-regression-techniques/test.csv"
stringless_training_data_path = "house-prices-advanced-regression-techniques/stringless_train.csv"
stringless_testing_data_path = "house-prices-advanced-regression-techniques/stringless_test.csv"
export_data_path = "Exported_Data/"

def get_training_data():
    """
    gets the initial training data from the specified folder.
    :return: Returns a dataframe of the training data.
    """
    training_data = pd.read_csv(training_data_path, index_col="Id")
    return training_data


def get_testing_data():
    """
    gets the initial testing data from the specified folder.
    :return: Returns a dataframe of the testing data.
    """
    testing_data = pd.read_csv(testing_data_path, index_col="Id")
    return testing_data


def get_stringless_training_data():
    """
    gets the training data without strings, as exported from file 1.1 from the specified folder.
    :return: Returns a dataframe of the training data.
    """
    training_data = pd.read_csv(stringless_training_data_path)
    return training_data


def get_stringless_testing_data():
    """
    gets the testing data without strings, as exported from file 1.1 from the specified folder.
    :return: Returns a dataframe of the testing data.
    """
    testing_data = pd.read_csv(stringless_testing_data_path)
    return testing_data


def clean_data(data):
    """
    Removes columns that have strings, and replaces remaining NaN's with 0.
    :param data: a dataframe
    :return: a cleaned version of the dataframe. Is mainly used in notebook 1.1 for initial cleaning.
    """
    column_to_remove = list(data.select_dtypes(include=['category', 'object']))
    cleaned_data = data.drop(column_to_remove, axis=1)
    cleaned_data = cleaned_data.replace({np.NaN: 0})
    return cleaned_data


def merge_predicted_price(data, prediction):
    """
    Merges the predicted price onto the X_test dataframe so that it can be repassed through the training models.
    :param data: a dataframe.
    :param prediction: a dataframe with the predicted price.
    :return: returns the dataframe data with an additional column.
    """
    predicted_list = pd.DataFrame(prediction, columns=['SalePrice'])
    merged_data = data.reset_index()
    merged_data["SalePrice"] = predicted_list
    merged_data.drop(['Id'], axis=1, inplace=True)
    merged_data.drop(['index'], axis=1, inplace=True)
    return merged_data


def plot_matrix(data, threshold, is_saved=False, filename=""):
    """
    Plots a correlation matrix.
    :param data: a dataframe
    :param threshold: a value between 0 and 1 to reduce the amount of entries in the matrix
    :param is_saved: True or False, saves the matrix as a png
    :param filename: string, changes the file name of the exported png
    :return: a correlation trimmed correlation matrix with values only above the threshold
    """
    matrix = data.corr().abs()
    columns_to_keep = matrix.iloc[-1, :] >= threshold
    rows_to_keep = matrix.iloc[:, -1] >= threshold
    new_matrix = matrix.loc[rows_to_keep, columns_to_keep]
    print(new_matrix.shape)
    plt.figure(figsize=(20, 10))
    sns.heatmap(new_matrix, annot=True)
    if is_saved:
        plt.savefig(f"{export_data_path}{filename}.png", dpi=600)

    return new_matrix


def plot_feature_importance(x_val, regressor, x_lim=20, y_lim=0.6, is_saved=False, filename="", fig_size=(20, 10)):
    """
    Bar chart that plots the feature importance as dictated by the regression function.
    :param x_val: Dataframe, X_train or X_test
    :param regressor: regression model ["forest", "gbr", "ols", "xgb"]
    :param x_lim: int, max number of columns to be graphed
    :param y_lim: float, between 0 and 1, changes height of graph
    :param is_saved: True or False, saves the graph as a png
    :param filename: string, changes the name of exported png
    :param fig_size: (int, int), changes the size of the figure
    :return: prints a bar chart with feature importance
    """
    feature_names = [x for x in x_val.columns]
    importance = regressor.feature_importances_
    sorted_lists = sorted(zip(importance, feature_names), reverse=True)
    importance, feature_names = [[x[0] for x in sorted_lists], [x[1] for x in sorted_lists]]
    plt.figure(figsize=fig_size)
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45)
    plt.ylabel(r"feature importance")
    plt.ylim(0, y_lim)
    plt.xlim(0, x_lim)
    plt.tight_layout()
    if is_saved:
        plt.savefig(f"{export_data_path}{filename}.png", dpi=600)
    plt.show()


def plot_multi_scatter(columns, y_feature_name, data, is_saved=False, filename="", column_len=3, row_len=2, fig_size=(20,10)):
    """

    :param columns:
    :param y_feature_name:
    :param data:
    :param is_saved:
    :param filename:
    :param column_len:
    :param row_len:
    :param fig_size:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    #column_len = math.ceil(len(columns) / 2)
    x = data.loc[:, columns]
    y = data[y_feature_name]
    x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=columns)
    fig, axs = plt.subplots(ncols=column_len, nrows=row_len, figsize=fig_size)
    index = 0
    axs = axs.flatten()
    for i, k in enumerate(columns):
        sns.regplot(y=y, x=x[k], ax=axs[i])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    if is_saved:
        plt.savefig(f"{export_data_path}{filename}.png", dpi=600)


def plot_observed_vs_predicted(observed=None, predicted=None, x_lim=100, y_lim=500000, is_saved=False, filename="", size_of_fig=None):
    """
    Line chart showing the original vs predicted pricing.
    :param observed: dataframe, y_train
    :param predicted: dataframe, y_predicted, y_train_predicted
    :param x_lim: int, changes sample size
    :param y_lim: int, changes height of graph
    :param is_saved: True or False,  checks if graph should be saved
    :param filename: changes filename on export
    :param size_of_fig: (int, int) changes figure size
    :return: prints a line chart
    """
    fig, ax = plt.subplots(figsize=size_of_fig)
    if observed is not None:
        plt.plot(observed, alpha=0.8, label=r"Observed Price")
    if predicted is not None:
        plt.plot(predicted, alpha=0.8, label=r"Predicted Price")
    # always label your axes
    plt.xlabel(r"Sample Number")
    plt.ylabel(r"House Price")
    # create a legend
    plt.legend(loc="upper left")
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.tight_layout()
    if is_saved:
        plt.savefig(f"{export_data_path}{filename}.png", dpi=600)
    plt.show()


def plot_single_scatter(observed, predicted, x_lim=500000, y_lim=500000, is_saved=False, filename="", size_of_fig=None):
    """
    Scatter plot for a single prediction vs observed pricing.
    :param observed: dataframe, y_train
    :param predicted: dataframe, y_predicted, y_train_predicted
    :param x_lim: int, changes x value of graph
    :param y_lim: int, changes y value of graph
    :param is_saved: True or False,  checks if graph should be saved
    :param filename: changes filename on export
    :param size_of_fig: (int, int) changes figure size
    :return: prints a scatter plot
    """
    fig, ax = plt.subplots(figsize=size_of_fig)
    plt.plot(predicted, observed, ".", alpha=0.6)
    plt.xlabel(r"Predicted Price")
    plt.ylabel(r"Observed Price")
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.tight_layout()
    if is_saved:
        plt.savefig(f"{export_data_path}{filename}.png", dpi=600)
    plt.show()

