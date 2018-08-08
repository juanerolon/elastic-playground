# Collection of utilities and dataframe manipulation methods for Capstone Project
# Udacity's Machie Learning Nanodegree Certification
# @Juan E. Rolon
# https://github.com/juanerolon

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot
import feather

from sklearn.preprocessing import binarize

def weighted_Sum(df, features, weights, flabel,normalize=True):
    """
    Returns the weighted sum of selected features in dataframe
    :param df: input dataframe
    :param features: selected features to be weighted
    :param weights: list of weight coefficients
    :param flabel: (str) label name of feature returned by function
    :return: returns engineered feature frame[flabel] single pandas series (df column)
    """

    frame = df.copy(deep=True)

    for m, feat in enumerate(features):
        frame[feat] = weights[m] * frame[feat]

    frame[flabel] = frame.sum(axis=1)

    if normalize:
        cnorm = np.sum(weights)
        if (cnorm == 0.0):
            raise Exception("Can't operate with zero weights")
        s = frame[flabel] / cnorm
        return s.to_frame()
    else:
        s = frame[flabel]
        return s.to_frame()


def create_binarized_df(input_df, features, thresholds, full_binarization=True):
    """
    Binarizes selected features in dataframe according to some thresholding criterion

    :param input_df: dataframe containing the features to be binarized
    :param features: list of features to be binarized
    :param thresholds: threshold values to be passed to sklearn's binarize preprocessing method
    :param full_binarization: boolean value that sets whether we return a version of input_df with all its features
    binarized (True) or a partial dataframe containing the binarized version of the specified
    features in the features list (False).
    :return:
    """


    if full_binarization:
        if (len(features) < len(input_df.columns)):
            raise Exception("The list of input features must contain all features in dataframe"
                            "e.g. input_df.columns.")
        if (len(features) > len(input_df.columns)):
            raise Exception("The number of features in features list must be less than or equal to"
                            "the number of all features in input dataframe.")


        frame = input_df.copy(deep=True)
    else:
        if (len(features) > len(input_df.columns)):
            raise Exception("The number of features in features list must be less than or equal to"
                            "the number of all features in input dataframe.")
        frame = input_df[features]

    for feat in features:
        binarize(frame[feat].values.reshape(-1,1), threshold=thresholds[feat], copy=False)
        frame[feat] = pd.to_numeric(frame[feat], downcast='integer')

    return frame



def filter_outliers(input_data, LB=25.0, UP=75.0):
    import collections
    """Performs a hard removal of outliers in all columns of dataframe
    following Tukey's criterion. WARNING: It may impact data considerably.
    If needed, look for alternative methods if outliers exhibit non-trivial
    distributions. Here LB and UP define the bounds such that data points outside these
    bounds are considered outliers"""

    # For each feature find the data points with extreme high or low values
    outliers = []
    for feature in input_data.keys():
        Q1 = np.percentile(input_data[feature], LB) #25
        Q3 = np.percentile(input_data[feature], UP) #75
        step = 1.5 * (Q3 - Q1)

        feat_outliers = input_data[~((input_data[feature] >= Q1 - step) & (input_data[feature] <= Q3 + step))]
        outliers += list(feat_outliers.index.values)

    # Remove the outliers, if any were specified
    outliers = list(np.unique(np.asarray(outliers)))
    print(len(outliers))
    #good_data = input_data.drop(input_data.index[outliers]).reset_index(drop=True)
    good_data = None

    return good_data


def bitwise_index_compare(df1, df2,return_flag=False):
    """Compare indexes among two dataframes.
    Optionally returns the following set operations
     df1 - df2: set of indexes belonging exclusively to df1
     df2 -df1 : set of indexes belonging exclusively to df2
     intersection(df2,df1): common indexes to df1 and df2
     """

    df1_minus_df2 = np.setdiff1d(df1.index, df2.index)
    df1_intersection_df2 = np.intersect1d(df1.index, df2.index)
    df2_minus_df1 = np.setdiff1d(df2.index, df2.index)

    if return_flag:
        return df1_minus_df2, df1_intersection_df2, df2_minus_df1
    else:
        print("\n\nThere are exclusively {} index elements in Index set 1 respect to Index set 2".format(
            len(df1_minus_df2)))
        print("There are {} index elements in common to the supplied dataframes ".format(
            len(df1_intersection_df2)))
        print(
            "There are exclusively {} index elements in Index set 2 respect to Index set 1".format(len(df2_minus_df1)))
        return None



def bit_logic(x, y, op):
    """Implements bitwise boolean operations among bits x and y
    Bit states are represented by integers 0,1"""
    if op == 'OR':
        return int((x or y))
    elif op == 'AND':
        return int((x and y))
    elif op == 'NAND':
        return int(not (x and y))
    elif op == 'NOR':
        return int(not (x or y))
    elif op == 'XOR':
        return int((x and (not y)) or ((not x) and y))
    else:
        raise Exception("Incorrect Boolean operator selection")


def switch_df_index(df, feature):
    """Swithc original dataframe index to be unique values specified by
    the column values in feature"""
    df[feature] = pd.to_numeric(df[feature], downcast='integer')
    dflm = df.set_index(feature)
    return dflm


def restrict_by_interval(df, feature, min_val, max_val, boundary):
    """Restricts daframe to records containing values inside the interval specified
    for a continuous feature.
    Inputs:
    dataframe: df
    continuous feature: feature
    lower limit: min_val
    upper limit: max_val
    boundary or interval type: boundary: 'inclusive', 'exclusive', 'left', 'right'
    """

    if (boundary == 'inclusive'):
        dflm = df[(df[feature] >= min_val)]
        dflm = dflm[dflm[feature] <= max_val]
        return dflm
    elif (boundary == 'exclusive'):
        dflm = df[(df[feature] >= min_val)]
        dflm = dflm[dflm[feature] <= max_val]
        return dflm
    elif (boundary == 'left'):
        dflm = df[(df[feature] >= min_val)]
        dflm = dflm[dflm[feature] < max_val]
        return dflm
    elif (boundary == 'right'):
        dflm = df[(df[feature] > min_val)]
        dflm = dflm[dflm[feature] <= max_val]
        return dflm
    else:
        raise Exception("Incorrect interval boundary specification.\n"
                        "Choose between 'inclusive []', 'exclusive ()', 'left [)', 'right (]'")
        return None


def describe_df(df, n=2, cols=True, header=True, descr=False):
    """Provides a description of the dataframe. Includes list of
    column names, dataframe head and descriptive stats description"""
    if cols:
        print("\nDataframe column names:\n")
        for ct in df.columns:
            print(ct)
    if header:
        print("\nDataframe head:\n")
        print(df.head(n))
    if descr:
        print("\nDataframe description:\n")
        print(df.describe())


def get_feature_counts(df, features):
    """Provides a record's count of all the unique values belonging to a
    given datframe (df) feature. NOTE: features is a sequence input"""
    for feat in features:

        print("\nProcessing feature {}\n".format(feat))
        unique_elems = np.sort(df[feat].unique())
        snv = 0
        for val in np.sort(unique_elems):
            nv = df[df[feat] == val][feat].count()
            snv += nv
            print("No. of records of {} == {} : {}".format(feat, val, nv))
        print('Count total: {}\n'.format(snv))


def count_feature_nans(df, features):

    for feat in features:
        print("\nProcessing feature {}\n".format(feat))
        nan_count = df[feat].isnull().sum()
        feat_count = df[feat].count()
        total = feat_count + nan_count
        nans_percentage = np.round((nan_count/total) * 100.0, 2)
        print('NaN count: {}  ({} % of data series)\n'.format(nan_count, nans_percentage))

def count_rows_with_nans(df):
    dfn = df[df.isnull().any(axis=1)]
    tot_nan_rows = len(dfn.index)
    tot_num_rows = len(df.index)
    drop_nan_pct = np.round((tot_nan_rows/tot_num_rows * 100.0),2)
    print('Number of rows with NaNs: {}  ({} % of {} dataframe rows)\n'.format(tot_nan_rows, drop_nan_pct,tot_num_rows))

def get_nanrows_indexes(df):
    dfn = df[df.isnull().any(axis=1)]
    return list(dfn.index)


def headcounts(df,limit=5):
    print(df.head(limit))
    print("\nCounts:")
    print(df.count())


def save_KerasHistory_metrics(h_scratch):
    # Save history to CSV file
    history_data = pd.DataFrame(h_scratch.history)
    history_data.to_csv(filename + '.csv')









