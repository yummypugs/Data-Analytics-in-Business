import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn import preprocessing
import math

training_data_path = "house-prices-advanced-regression-techniques/train.csv"
testing_data_path = "house-prices-advanced-regression-techniques/test.csv"
stringless_training_data_path = "house-prices-advanced-regression-techniques/stringless_train.csv"
stringless_testing_data_path = "house-prices-advanced-regression-techniques/stringless_test.csv"
export_data_path = "Exported_Data/"

def get_training_data():
    training_data = pd.read_csv(training_data_path, index_col="Id")
    return training_data


def get_testing_data():
    testing_data = pd.read_csv(testing_data_path, index_col="Id")
    return testing_data


def get_stringless_training_data():
    training_data = pd.read_csv(stringless_training_data_path)
    return training_data


def get_stringless_testing_data():
    testing_data = pd.read_csv(stringless_testing_data_path)
    return testing_data


def clean_data(data):
    column_to_remove = list(data.select_dtypes(include=['category', 'object']))
    cleaned_data = data.drop(column_to_remove, axis=1)
    cleaned_data = cleaned_data.replace({np.NaN: 0})

    return cleaned_data


def merge_predicted_price(data, prediction):
    predicted_list = pd.DataFrame(prediction, columns=['SalePrice'])
    merged_data = data.reset_index()
    merged_data["SalePrice"] = predicted_list
    merged_data.drop(['Id'], axis=1, inplace=True)
    merged_data.drop(['index'], axis=1, inplace=True)
    return merged_data


def plot_matrix(data, threshold, is_saved=False, filename=""):
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


def plot_feature_importance(x_val, forest, x_lim=20, y_lim=0.6, is_saved=False, filename="", fig_size=(20,10)):
    feature_names = [x for x in x_val.columns]
    importance = forest.feature_importances_
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

