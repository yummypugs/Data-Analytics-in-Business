import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns


def get_training_data():
    training_data = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
    return training_data


def get_testing_data():
    testing_data = pd.read_csv("house-prices-advanced-regression-techniques/test.csv")
    return testing_data


def run_random_forest(X_val, Y_val):
    forest = RandomForestRegressor()
    forest.fit(X_val, Y_val)

    return forest


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
    return merged_data


def plot_matrix(data, threshold):
    matrix = data.corr().abs()
    columns_to_keep = matrix.iloc[-1, :] >= threshold
    rows_to_keep = matrix.iloc[:, -1] >= threshold
    new_matrix = matrix.loc[rows_to_keep, columns_to_keep]
    new_matrix.shape
    plt.figure(figsize=(20, 10))
    sns.heatmap(new_matrix, annot=True)


def plot_feature_importance(x_val, forest, x_lim, y_lim):
    feature_names = [x for x in x_val.columns]
    importance = forest.feature_importances_
    sorted_lists = sorted(zip(importance, feature_names), reverse=True)
    importance, feature_names = [[x[0] for x in sorted_lists], [x[1] for x in sorted_lists]]
    plt.figure()
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45)
    plt.ylabel(r"feature importance")
    plt.ylim(0, y_lim)
    plt.xlim(0, x_lim)
    plt.tight_layout()
    plt.show()