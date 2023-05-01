import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib.colors import ListedColormap

def run_svm(sm_csv, ca_csv, check_sm_csv, check_ca_csv, kernel_function="linear",
            do_scatter=False, scatter_file="default_scatter.png",
            save_scatter=False):
    sm_df = pd.read_csv(sm_csv)
    ca_df = pd.read_csv(ca_csv)
    sm_test_df = pd.read_csv(check_sm_csv)
    ca_test_df = pd.read_csv(check_ca_csv) 
    
    sm_array = sm_df.to_numpy()
    ca_array = ca_df["Class"].to_numpy()
    sm_test_array = sm_test_df.to_numpy()
    ca_test_array = ca_test_df["Class"].to_numpy()
    
    clf = svm.SVC(kernel=kernel_function)
    clf.fit(sm_array, ca_array)
    
    score = clf.score(sm_test_array, ca_test_array)
    
    if do_scatter==True:
        full_df = sm_df
        full_df["Class"] = ca_array
        pair_scatter = sns.pairplot(data=full_df, diag_kind="hist", hue="Class", \
                                    palette={0:(1,0.7,0.2), 1:(0.2,0.2,1)}, \
                                    markers="o", plot_kws = dict(linewidth=0, alpha=0.8))
        if save_scatter==True:
            pair_scatter.savefig(scatter_file)
    print(kernel_function)
    print(score)
    return clf, score

    
def svm_pre_read(ca_train, sm_train, kernel_function="linear"):
    clf = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel_function))
    clf.fit(sm_train, ca_train)
    return clf


def pairwise_training(ca_train_file, sm_train_file, ca_test_file, sm_test_file, \
                      save_plot_to, kernel_function="linear", save_plot=True):
    ca_train_df = pd.read_csv(ca_train_file)
    sm_train_df = pd.read_csv(sm_train_file)
    ca_test_df = pd.read_csv(ca_test_file)
    sm_test_df = pd.read_csv(sm_test_file)
    
    ca_train = 1*ca_train_df["Class"].to_numpy()
    sm_train = 10000*sm_train_df.to_numpy()
    ca_test = 1*ca_test_df["Class"].to_numpy()
    sm_test = 10000*sm_test_df.to_numpy()
    
    parameter_list = sm_train_df.columns.values.tolist()
    
    no_features = len(parameter_list)
    boundary_cm = plt.cm.RdBu
    scatter_colours = ListedColormap(["red", "blue"])
    figure = plt.figure(figsize=(30, 25))
    for i in range(no_features):
        # only plot upper triangle
        # each row will share a y axis (column i)
        # each column will share an x axis (column j)
        for j in range(i+1, no_features):
            clf = svm_pre_read(ca_train, sm_train[:,[j,i]], kernel_function=kernel_function)
            score = clf.score(sm_test[:,[j,i]], ca_test)
            plot_index = (i*(no_features-1)) + j
            ax = plt.subplot((no_features-1), (no_features-1), plot_index)
            DecisionBoundaryDisplay.from_estimator(clf, sm_train[:,[j,i]], cmap=boundary_cm, ax=ax, alpha=0.8)
            ax.scatter(sm_train[:,j], sm_train[:,i], marker="o", c=ca_train, \
                       edgecolors="white", linewidths=0.5, cmap=scatter_colours, alpha=0.8)
            ax.text(0.65,0.05,score, fontsize=20, transform=ax.transAxes)
            
            if i == 0:
                ax.set_title(parameter_list[j], fontsize=25)
            if (j==(no_features-1)):
                ax.text(1.1, 0.5, parameter_list[i], fontsize=25, transform=ax.transAxes)
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot_to)
