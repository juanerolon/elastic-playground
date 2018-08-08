# Collection of graphic's utilities and dataframe manipulation methods for Capstone Project
# Udacity's Machine Learning Nanodegree Certification
# @Juan E. Rolon
# https://github.com/juanerolon

import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import feather


def hist_density_plots_bc(bin_df, cont_df, bin_feature, cont_feature, cont_label, mean_mark=True, median_mark=True):

    """
    Generates a four-panel plot of histograms and corresponding density function
    of the distribution of a continuous feature for the case in which the coupled
    binary feature attains a value of 1 (present) and 0 (not present), respectively.
    We assume a significant correlation between the binary and continuous feature.

    :param bin_df: dataframe containing the binary feature bin_feature
    :param cont_df: dataframe containing the continuous feature cont_feature
    :param bin_feature: the binary feature
    :param cont_feature: the continuous feature
    :param cont_label: plot x-label describing the continuous feature
    :return: None
    """

    df = pd.concat([bin_df[bin_feature], cont_df[cont_feature]], axis=1)
    dflm_p = df[(df[bin_feature] == 1)]
    dflm_n = df[(df[bin_feature] == 0)]

    plt.figure(1, figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title('Individuals Presenting ' + bin_feature, fontsize=10)
    dflm_p[cont_feature].plot(kind='hist', histtype='stepfilled', alpha=0.5, bins=50)

    plt.subplot(2, 2, 2)
    plt.title('Patients NOT Presenting ' + bin_feature, fontsize=10)
    dflm_n[cont_feature].plot(kind='hist', histtype='stepfilled', alpha=0.5, bins=50)

    plt.subplot(2, 2, 3)
    dflm_p[cont_feature].plot(kind='kde')
    plt.xlabel(cont_label)
    if mean_mark:
        plt.axvline(dflm_p[cont_feature].mean(), color='r', linestyle='dashed', linewidth=1);
    if median_mark:
        plt.axvline(dflm_p[cont_feature].median(), color='g', linestyle='dashed', linewidth=1);

    plt.subplot(2, 2, 4)
    dflm_n[cont_feature].plot(kind='kde')
    plt.xlabel(cont_label)
    if mean_mark:
        plt.axvline(dflm_n[cont_feature].mean(), color='r', linestyle='dashed', linewidth=1);
        plt.legend("Median", loc='best');
    if median_mark:
        plt.axvline(dflm_n[cont_feature].median(), color='g', linestyle='dashed', linewidth=1);


    plt.tight_layout()
    plt.show()

    return None



def plot_binary_histogram(df, feature, title, xlabel, ylabel,
                          ytrue_label, yfalse_label):
    """Creates a bar plot of a single feature that takes binary values (0, 1).
    input: dataframe(df), feature string name (feature)"""

    def gcd(a, b):
        """Compute greater common divisor to be used to express class
        ratio in simple fraction (if possible)"""
        while b:
            a, b = b, a % b
        return a


    s1 = df[df[feature] == 1][feature].count()
    s0 = df[df[feature] == 0][feature].count()
    bars = [s1, s0]
    s1_array = [s1]
    s0_array = [s0]

    maxval = np.max(bars)
    minval = np.min(bars)
    gcdval = gcd(maxval,minval)

    cmax = int(maxval/gcdval)
    cmin = int(minval/gcdval)
    ratio_legend = "Class ratio {}:{}".format(cmax, cmin)

    index = np.arange(len([1]))
    bar_width = 0.1
    opacity = 0.8

    plt.bar(index, s0_array, bar_width, alpha=opacity, color='b', label=yfalse_label)
    plt.bar(index + bar_width, s1_array, bar_width, alpha=opacity, color='r', label=ytrue_label)

    plt.ylim(0, maxval*1.25)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off


    plt.legend(frameon=False, loc='upper right', fontsize='small')
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    plt.text(-0.05, 1.1*maxval, ratio_legend, fontsize=10)


def twofeat_barplot(df, xfeature, yfeature, nbins, title,
                    xlabel, ylabel, ytrue_label, yfalse_label,xticks_rotation='horizontal',
                    false_bars=True, true_bars=True, verbose=False):

    """Plots a barplot histogram-like of the number of records corresponding to True and False values of a
    dataframe column 'yfeature' versus a continuous-valued column 'xfeature'. The x-axis bins are computed
    automatically after specifying the number of bins requested at input. The chart contains two bars per
    bin, corresponding each to a boolean True (0) or False (1).

    inputs:

    df: dataframe
    xfeature: string name of the column specified as x-axis in the plot
    yfeature: string name of the column speficied ax y-axis in the plot. yfeature values are binary (1 or 0)
    nbins: number of bins for the barplot.
    title: chart title
    xlabel: x-axis label
    ylabel: y-axis label
    ytrue_label: label for bar assigned to number of True values in yfeature column
    yfalse_label: label for bar assigned to number of False values in yfeature column
    xticks_rotation: orientation of bin labels as specified by matplotlib
    verbose: if True prints summary counts info for each dataframe used to plot bars generated per bin

    """

    max = np.max(df[xfeature].values)
    min = np.min(df[xfeature].values)
    rng = np.abs(max-min)
    bin_size = np.round(rng/nbins,0)

    #Create bins
    bins = []
    bins_str = []

    span = np.floor(min)

    while span <= np.ceil(max) + bin_size:
        bins.append(span)
        span += bin_size

    bins = np.round(bins, 2)

    for m in range(len(bins) -1 ):
        label = "{} to {}".format(bins[m], bins[m+1])
        bins_str.append(label)

    #Fill bars over bins
    s1_array = []
    s0_array = []

    for m in range(len(bins)-1):
        df1 = df[(df[xfeature] >= bins[m])]
        df2 = df1[df1[xfeature] < bins[m+1]]
        nv =  df2[xfeature].count()

        s1 = df2[df2[yfeature] == 1][yfeature].count()
        s0 = df2[df2[yfeature] == 0][yfeature].count()
        s1_array.append(s1)
        s0_array.append(s0)

        if verbose:
            print("Dataframe Bin")
            print(df2)
            print("Bin: " + bins_str[m])
            print("Sample count: {}".format(nv))
            print("")
            print("True counts s1 = {}, False counts s0={}".format(s1,s0))
            print("")


    index = np.arange(len(bins_str))
    bar_width = 0.35
    opacity = 0.8

    if false_bars:
        plt.bar(index, s0_array, bar_width, alpha=opacity, color='k', label=yfalse_label)
    if true_bars:
        plt.bar(index + bar_width, s1_array, bar_width, alpha=opacity, color='g', label=ytrue_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width / 2.0, bins_str, rotation=xticks_rotation)
    plt.legend(frameon=False, loc='upper right', fontsize='small')


def plot_num_featHistogram(df, feature):
    """Plots a histogram of a single dataframe feature with numeric type.
    Uses a bin size equal to the number of unique values of the feature column"""
    x = df[feature].values
    nbins = len(list(df[feature].unique()))
    plt.title("{} Feature Distribution".format(feature),fontsize = 14)
    plt.xlabel("{}".format(feature))
    plt.ylabel("Number of Records")
    plt.hist(x, bins=nbins, facecolor='green', edgecolor='black', alpha=0.75)

    plt.tight_layout()
    plt.show()


def plot_features_bars(df, features, figsize=(16, 16), nrows=2, ncols=2):
    """Plots a set of bar plots. A graph is generate per feature.
    A bar is generated per each unique value on a given feature.
    It is assumed that the feature has been split into categorical
    bins"""
    plot_num = 1
    plt.figure(1, figsize=figsize)

    for feat in features:
        groups = df[feat].unique() #
        gelems = len(groups)
        bars_array = [0] * gelems

        for i, val in enumerate(groups):
            bars_array[i] = df[df[feat] == val][feat].count()

        plt.subplot(nrows, ncols, plot_num)
        index = np.arange(gelems)
        bar_width = 0.35
        opacity = 0.8

        plt.bar(index, bars_array, bar_width, alpha=opacity, color='gray', label='{}'.format(val))

        plt.ylabel('No. Records')
        plt.title('Feature: {}'.format(feat))
        plt.xticks(index, groups, rotation='vertical')
        plot_num += 1

    plt.tight_layout()
    plt.show()

def exp_variance_plots(input_data, n_plots, n_setbars):
    import seaborn as sns
    from sklearn.decomposition import PCA
    sns.reset_orig()

    if (n_plots*n_setbars) != len(input_data.keys()):
        raise Exception("Number of plots x Number of Bar subplots needs to equal number of pca dimensions")

    pca = PCA(n_components=input_data.shape[1])
    pca.fit(input_data)

    # PCA components
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]
    components = pd.DataFrame(np.round(pca.components_, 4), columns=input_data.keys())
    components.index = dimensions

    # PCA explained variance ratios
    bare_ratios = pca.explained_variance_ratio_

    for i in range(n_plots):

        fig, ax = plt.subplots(figsize=(14, 8))
        sc = components.loc[components.index[n_setbars * i:n_setbars * (i + 1)]]
        vr = bare_ratios[n_setbars * i:n_setbars * (i + 1)]
        sc.plot(ax=ax, kind='bar');
        ax.set_ylabel("Feature Weights")
        ax.set_xticklabels(sc.index, rotation=0)

        for j, ev in enumerate(vr):
            ax.text(j - 0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f" % (ev))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.show()


def pca_results(good_data, pca):
	'''
	Constructs barchart plots of principal components analysis of
	the input dataframe "good_data" given the input pca object "pca".
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)


def biplot(good_data, reduced_data, pca):
    '''
    Note: it works for two components only!!
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.

    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)

    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    import seaborn as sns
    sns.reset_orig()

    fig, ax = plt.subplots(figsize=(14, 8))
    # scatterplot of the reduced data
    ax.set_axis_bgcolor('gray')
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'],
               facecolors=None, edgecolors='b', s=70, alpha=0.5)

    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size * v[0], arrow_size * v[1],
                 head_width=0.2, head_length=0.2, linewidth=2, color='yellow')
        ax.text(v[0] * text_pos, v[1] * text_pos, good_data.columns[i], color='red',
                ha='center', va='center', fontsize=12)

    ax.set_ylim([-4.0, 7.0])
    ax.set_xlim([-4.0, 6.0])

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax


def plot_KerasHistory_metrics(h, filename=None, show_figure=False):

    """Plots the Keras Model Training history.
     Input: (object) h = Keras history object
     Input: (str)    filename = Optional filename for saving history into a PNG file
     Input:(boolean) show_figure = Whether to print show figure in screen
     """



    plt.figure(1, figsize=(10, 4))

    plt.subplot(1, 2, 1)
    # summarize history for accuracy
    plt.plot(h.history['acc'], color='b')
    plt.plot(h.history['val_acc'], color='k')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    # summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save and show training and validation performance metrics plots
    plt.savefig(filename + '.png', dpi=300, orientation='landscape')

    if show_figure:
        plt.show()













