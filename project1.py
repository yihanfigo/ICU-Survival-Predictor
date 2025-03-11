"""
EECS 445 Winter 2025

This script should contain most of the work for the project. You will need to fill in every TODO comment.
"""


import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer
import helper


__all__ = [
    "generate_feature_vector",
    "impute_missing_values",
    "normalize_feature_matrix",
    "get_classifier",
    "performance",
    "cv_performance",
    "select_param_logreg",
    "select_param_RBF",
    "plot_weight",
]


# load configuration for the project, specifying the random seed and variable types
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
np.random.seed(seed)


def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.

    Args:
        df: pd.Dataframe, with columns [Time, Variable, Value]

    Returns:
        a python dictionary of format {feature_name: feature_value}
        for example, {"Age": 32, "Gender": 0, "max_HR": 84, ...}
    """
    
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # TODO: 1) Replace unknown values with np.nan
    # NOTE: pd.DataFrame.replace() may be helpful here, refer to documentation for details
    df_replaced = df.replace(-1, np.nan)

    # Extract time-invariant and time-varying features (look into documentation for pd.DataFrame.iloc)
    static, timeseries = df_replaced.iloc[0:5], df_replaced.iloc[5:]

    feature_dict = {}
    # TODO: 2) extract raw values of time-invariant variables into feature dict
    for _,row in static.iterrows():
        feature_dict[row.iloc[1]] = row.iloc[2]
    # TODO  3) extract max of time-varying variables into feature dict
    for _, row in timeseries.iterrows():
        key = row.iloc[1]
        nkey = f"max_{key}"
        value = row.iloc[2]
        if nkey not in feature_dict:
            feature_dict[nkey] = value
        else:
            feature_dict[nkey] = max(value, feature_dict[nkey])
    for v in timeseries_variables:
        nv = f"max_{v}"
        if nv not in feature_dict:
            feature_dict[nv] = np.nan
    return feature_dict
    """
    #code for challenge
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # 1) Replace unknown values with np.nan
    df_replaced = df.replace(-1, np.nan)

    # Extract time-invariant (first 5 rows) and time-varying features (remaining rows)
    static = df_replaced.iloc[0:5]
    timeseries = df_replaced.iloc[5:]

    feature_dict = {}
    # 2) Extract raw values of time-invariant variables into feature_dict.
    # Change the way we treated ICUType (convert to four one-hot encoded features)
    for _, row in static.iterrows():
        variable_name = row.iloc[1]
        value = row.iloc[2]
        if variable_name == "ICUType":
            # Convert ICUType numeric value into four one-hot encoded features.
            icu_mapping = {
                1: "Coronary Care Unit",
                2: "Cardiac Surgery Recovery Unit",
                3: "Medical ICU",
                4: "Surgical ICU"
            }
            # Initialize all ICU type features to 0.
            for icu_feature in icu_mapping.values():
                feature_dict[icu_feature] = 0
            # If the ICUType value is valid, set the corresponding feature to 1.
            if pd.notna(value) and value in icu_mapping:
                feature_dict[icu_mapping[value]] = 1
        else:
            feature_dict[variable_name] = value

    # 3) For each timeseries variable, compute median, max, and min.
    #    Use the 24-48 hour window if available; otherwise, use the 0-24 hour window.
    for var in timeseries_variables:
        # Filter rows for the current variable.
        var_data = timeseries[timeseries["Variable"] == var].copy()
        
        if not var_data.empty:
            # Convert "Time" from "HH:MM" to numeric hours.
            var_data["Time"] = pd.to_timedelta(var_data["Time"] + ':00').dt.total_seconds() / 3600
            
            # Try to compute summary statistics from the 24-48 hour window first.
            window_24_48 = var_data[var_data["Time"] >= 24]
            if not window_24_48.empty:
                median_value = window_24_48["Value"].median()
                max_value = window_24_48["Value"].max()
                min_value = window_24_48["Value"].min()
            else:
                # If no measurements in 24-48, try the 0-24 hour window.
                window_0_24 = var_data[var_data["Time"] < 24]
                if not window_0_24.empty:
                    median_value = window_0_24["Value"].median()
                    max_value = window_0_24["Value"].max()
                    min_value = window_0_24["Value"].min()
                else:
                    median_value = np.nan
                    max_value = np.nan
                    min_value = np.nan
        else:
            median_value = np.nan
            max_value = np.nan
            min_value = np.nan

        # Store the computed values in the feature dictionary.
        feature_dict[f"median_{var}"] = median_value
        feature_dict[f"max_{var}"] = max_value
        feature_dict[f"min_{var}"] = min_value

    return feature_dict
    """
    

def impute_missing_values(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, impute missing values (np.nan) with the population mean for that feature.

    Args:
        X: array of shape (N, d) which could contain missing values
        
    Returns:
        X: array of shape (N, d) without missing values
    """
    
    # TODO: implement
    cols_mean = np.nanmean(X, axis = 0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i,j]):
                X[i,j] = cols_mean[j]

    return X
    """
    
    #code for challenge
    # Compute the median for each column, ignoring NaNs
    cols_median = np.nanmedian(X, axis=0)
    
    # Replace missing values with the median of the corresponding column
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i, j]):
                X[i, j] = cols_median[j]

    return X
    """
    


def normalize_feature_matrix(X: npt.NDArray) -> npt.NDArray:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: array of shape (N, d).

    Returns:
        X: array of shape (N, d). Values are normalized per column.
    """
    
    scaler = MinMaxScaler(feature_range=(0, 1))  # Default is [0, 1]
    X_normalized = scaler.fit_transform(X)  # Fit and transform data

    return X_normalized
    



def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization.
        C: Regularization strength parameter.
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression.
        gamma: Kernel coefficient.

    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(
            penalty=penalty,
            C=C,
            solver = "liblinear",
            class_weight=class_weight,
            fit_intercept=False,
            random_state=seed
        )

    elif loss == "squared_error":
        return KernelRidge(alpha=1.0 / (2 * C), kernel=kernel, gamma=gamma)


def performance(
    clf_trained: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y_true: npt.NDArray,
    metric: str = "accuracy"
) -> float:
    """
    Calculates the performance metric as evaluated on the true labels
    y_true versus the predicted scores from clf_trained and X.
    Returns single sample performance as specified by the user. Note: you may
    want to implement an additional helper function to reduce code redundancy.

    Args:
        clf_trained: a fitted instance of sklearn estimator
        X : (n,d) np.array containing features
        y_true: (n,) np.array containing true labels
        metric: string specifying the performance metric (default='accuracy'
                other options: 'precision', 'f1-score', 'auroc', 'average_precision',
                'sensitivity', and 'specificity')
    Returns:
        peformance for the specific metric
    """
    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    # TODO: implement
    if isinstance(clf_trained, KernelRidge):
        y_tmp = clf_trained.predict(X)
        y_prob = y_tmp
        y_pred = np.where(y_tmp >= 0, 1, -1)
    elif isinstance(clf_trained, LogisticRegression):
        y_prob = clf_trained.decision_function(X)
        y_pred = clf_trained.predict(X)
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred,labels = [-1,1],zero_division=0)
    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_pred, labels = [-1,1],zero_division= 0)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_prob,labels = [-1,1])
    elif metric == "average_precision":
        return metrics.average_precision_score(y_true, y_prob)
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred,labels = [-1,1],zero_division=0)
    elif metric == "specificity":
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred,labels=[-1,1]).ravel()
        if tn+fp == 0:
            return 0
        else:
            return tn/(tn + fp)
    else:
         raise ValueError(f"Unrecognized metric: {metric}")
        


def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.

    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')

    Returns:
        a tuple containing (mean, min, max) cross-validation performance across the k folds
    """
    # NOTE: you may find sklearn.model_selection.StratifiedKFold helpful
    # TODO: implement
    skf = StratifiedKFold(n_splits=k, shuffle= False)
    skf.get_n_splits(X,y)
    perf = np.zeros(k)
    i = 0
    for train_index, val_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]

        clf.fit(X_train, y_train)
        p = performance(clf, X_val, y_val,metric)
        perf[i] = p
        i+=1
    
    ma = np.nanmax(perf)
    mi = np.nanmin(perf)
    mean = np.nanmean(perf)
    
    return mean,mi,ma



def select_param_logreg(
    X: npt.NDArray,
    y: npt.NDArray,
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression, calculating the k-fold CV
    performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over

    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # NOTE: use your cv_performance function to evaluate the performance of each classifier
    # TODO: implement
    m_perf = -np.inf
    ma = -np.inf
    mi = -np.inf
    m_c = 0
    penalty = ""
    for c in C_range:
        clf1 = get_classifier(loss = "logistic", penalty="l1", C=c, class_weight= None)
        clf2 = get_classifier(loss = "logistic", penalty="l2", C=c, class_weight= None)
        mean1,min1,max1 = cv_performance(clf1, X, y, metric,k)
        mean2,min2,max2 = cv_performance(clf2, X, y, metric,k)
        
        if mean1 >= mean2:
            if mean1 > m_perf:
                m_perf = mean1
                ma = max1
                mi = min1
                m_c = c
                penalty = "l1"
        else:
            if mean2 > m_perf:
                m_perf = mean2
                ma = max2
                mi = min2
                m_c = c
                penalty = "l2"
    print(f"{metric} CV performance: {m_perf}({mi},{ma})\n")
    return m_c, penalty





def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.

    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over

    Returns:
        The parameter values for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    # NOTE: this function should be similar to your implementation of select_param_logreg
    # TODO: implement
    m_perf = -np.inf
    ma = -np.inf
    mi = -np.inf
    m_c = 0
    m_g = 0
    penalty = ""
    for c in C_range:
        for g in gamma_range:
            clf = KernelRidge(alpha=1.0/(2*c), kernel="rbf", gamma=g)
            mean,min,max = cv_performance(clf, X, y, metric,k)
            print(f"gamma: {g}, mean{mean}, min{min}, max{max}\n")
            if mean > m_perf:
                m_perf = mean
                ma = max
                mi = min
                m_c = c
                m_g = g
    
    return m_c, m_g

def plot_weight(
    X: npt.NDArray,
    y: npt.NDArray,
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}

    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")

    for penalty in penalties:
        norm0 = []
        for C in C_range:
            # TODO: initialize clf with C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", fit_intercept=False, random_state=seed)
            
            # TODO: fit clf to X and y
            clf.fit(X,y)
            # TODO: extract learned coefficients from clf into w
            # NOTE: the sklearn.linear_model.LogisticRegression documentation will be helpful here
            w = clf.coef_
            
            # TODO: count the number of nonzero coefficients and append the count to norm0
            non_zero_count = np.count_nonzero(w)
            norm0.append(non_zero_count)

        # This code will plot your L0-norm as a function of C
        plt.plot(C_range, norm0)
        plt.xscale("log")
    plt.legend([penalties[0], penalties[1]])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")

    plt.savefig("L0_Norm.png", dpi=200)
    plt.close()

def question1(
        X_train:npt.NDArray,
        feature_names:list[str]
): 
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    mean = np.mean(X_train, axis=0)
    q75 = np.percentile(X_train, 75, axis=0)
    q25 = np.percentile(X_train, 25, axis=0)
    inter = q75-q25    
    var = np.concatenate((static_variables, timeseries_variables))
    var = var.astype(object)
    for i in range(5,var.shape[0]):
        var[i] = f"max_{var[i]}"
    dtype = [('name','U50'),('mean','f8'),('inter','f8')]
    result = np.zeros(40,dtype)
    i=0
    for j in range(len(feature_names)):
        if feature_names[j] in var:
            result[i] = (feature_names[j], mean[j],inter[j])
            i+=1
    df = pd.DataFrame(result)
    df.to_csv("output.csv",index=False)
    print(result)
    
def question2(
        X:npt.NDArray,
        y:npt.NDArray,
        metric_list:list[str]    
              ):
    C = [0.001,0.01,0.1,1,10,100,1000]
    penalties = ["l2","l1"]
    mc_metric = []
    penalty_metric = []
    for metric in metric_list:
        mc, pe = select_param_logreg(X,y,metric,5,C,penalties)
        mc_metric.append(mc)
        penalty_metric.append(pe)
        print(f"C:{mc}. Penalty:{pe}\n")

def question2d(X:npt.NDArray,
        y:npt.NDArray,
        X_test:npt.NDArray,
        y_test:npt.NDArray,
        metric_list:list[str]):
    clf = LogisticRegression(
            penalty="l1",
            C=1,
            solver = "liblinear",
            fit_intercept=False,
            random_state=seed
        )
    clf.fit(X,y)
    for metric in metric_list:
        print(f"{metric}: {performance(clf, X_test, y_test, metric= metric)}")

    
    
def question2e(X:npt.NDArray,
        y:npt.NDArray):
    C = [0.001,0.01,0.1,1,10,100,1000]
    penalties = ["l2","l1"]
    plot_weight(X,y,C,penalties)

def question2f(X:npt.NDArray,
        y:npt.NDArray,
        feature_names:list[str]):
    clf = LogisticRegression(
            penalty="l1",
            C=1,
            solver = "liblinear",
            fit_intercept=False,
            random_state=seed
        )
    clf.fit(X,y)
    w = clf.coef_.flatten()
    feature_names = np.array(feature_names)  # Convert feature names list to a NumPy array

    
    mpi = np.argsort(w)[-4:] 
    mni = np.argsort(w)[:4]   

   
    mpf = feature_names[mpi]
    mpv = w[mpi]

    mnf = feature_names[mni]
    mnv = w[mni]

    
    print("Top 4 Positive Coefficients:")
    for name, value in zip(mpf, mpv):
        print(f"{name}: {value:.4f}")

    print("\nTop 4 Negative Coefficients:")
    for name, value in zip(mnf, mnv):
        print(f"{name}: {value:.4f}")

def question3b(X:npt.NDArray,
        y:npt.NDArray,
        X_test:npt.NDArray,
        y_test:npt.NDArray,
        metric_list:list[str]):
    
    clf = LogisticRegression(
            penalty="l2",
            C=1,
            solver = "liblinear",
            class_weight= {-1:1, 1:50},
            fit_intercept=False,
            random_state=seed
        )
    clf.fit(X,y)
    for metric in metric_list:
        print(f"{metric}: {performance(clf, X_test, y_test, metric= metric)}")

def question33(X:npt.NDArray,
        y:npt.NDArray,
        X_test:npt.NDArray,
        y_test:npt.NDArray,
        metric_list:list[str],wp,wn):
    
    clf = LogisticRegression(
            penalty="l2",
            C=1,
            solver = "liblinear",
            class_weight= {-1:wn, 1:wp},
            fit_intercept=False,
            random_state=seed
        )
    """
    #code for challenge
    clf = LogisticRegression(
            penalty="l2",
            C=1,
            solver = "liblinear",
            class_weight= {-1:wn, 1:wp},
            fit_intercept=False,
            random_state=seed
        )
    """
    clf.fit(X,y)
    for metric in metric_list:
        print(f"{metric}: {performance(clf, X_test, y_test, metric= metric)}")
    
def question3ROC(X:npt.NDArray,
        y:npt.NDArray,
        X_test:npt.NDArray,
        y_test:npt.NDArray):
        clf1 = LogisticRegression(
            penalty="l2",
            C=1,
            solver = "liblinear",
            class_weight= {-1:1, 1:5},
            fit_intercept=False,
            random_state=seed
        )
        clf1.fit(X,y)
        y_score1 = clf1.decision_function(X_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score1)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='b', label=f"{"Wn = 1, Wp = 5"} (AUC = {roc_auc:.2f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for Different Class Weights (C = 1.0)")
        plt.legend()
        plt.grid()
        plt.show()
def question41(X:npt.NDArray,
        y:npt.NDArray,
        X_test:npt.NDArray,
        y_test:npt.NDArray,
        metric_list:list[str]):
    C=1.0
    logclf = LogisticRegression(penalty="l2", C=C, fit_intercept=False, random_state=seed)
    kernelclf = KernelRidge(alpha=1/(2*C), kernel="linear")
    logclf.fit(X,y)
    print("Performance of logisticregression:\n")
    for metric in metric_list:
        print(f"{metric}: {performance(logclf, X_test, y_test, metric= metric)}")
    kernelclf.fit(X,y)
    print("Performance of kernelridge:\n")
    for metric in metric_list:
        print(f"{metric}: {performance(kernelclf, X_test, y_test, metric= metric)}")
    
def question42(X:npt.NDArray,
        y:npt.NDArray,
        X_test:npt.NDArray,
        y_test:npt.NDArray,
        metric_list:list[str]):
    gamma_range = [0.01,0.1,1,10]
    c_range = [0.01,0.1,1,10,100]
    max_c, max_gamma = select_param_RBF(X, y, metric = "auroc",k = 5, C_range = c_range, gamma_range=gamma_range)
    print(f"max_c:{max_c}, max_gamma:{max_gamma}")
    clf = KernelRidge(alpha=1.0/(2*max_c), kernel="rbf", gamma=max_gamma)
    clf.fit(X,y)
    for metric in metric_list:
        print(f"{metric}: {performance(clf, X_test, y_test, metric= metric)}")



    
    #for metric in metric_list:
        #print(f"{metric}: {performance(clf, X_test, y_test, metric= metric)}")
def main():
    print(f"Using Seed = {seed}")
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED IMPLEMENTING generate_feature_vector,
    #       fill_missing_values AND normalize_feature_matrix!
    # NOTE: Only set debug=True when testing your implementation against debug.txt. DO NOT USE debug=True when
    #       answering the project questions! It only loads a small sample (n = 100) of the data in debug mode,
    #       so your performance will be very bad when working with the debug data.
    #X_train, y_train, X_test, y_test, feature_names = helper.get_project_data(debug=False)
    #rus = RandomUnderSampler(random_state=seed)
    #X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)
    



    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]
    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    #       sub-question/question to organize your code!
    #question1(X_train, feature_names)
    #question2(X_train, y_train, metric_list)
    #question2d(X_train, y_train, X_test, y_test, metric_list)
    #question2e(X_train, y_train)
    #question2f(X_train, y_train, feature_names)
    #question3b(X_train, y_train, X_test, y_test, metric_list)
    """
    ncount = np.sum(y_train == -1)
    pcount = np.sum(y_train == 1)
    
    wp = (ncount + pcount)/(2*pcount)
    wn = (ncount+pcount)/(2*ncount)
    
    print(f"postive{pcount}, negative{ncount}. sum {ncount + pcount} \n")
    print(f"positive weight{1600.0/(2*pcount)}. Negative weight{1600.0/(2*ncount)}")
    """
    #question33(X_train, y_train, X_test, y_test, metric_list,wp,wn)
    
    #question3ROC(X_train,y_train, X_test, y_test)
    #question41(X_train, y_train, X_test, y_test, metric_list)
    #gamma_range = [0.001,0.01,0.1,1,10,100]
    """
    c_range = [1]
    max_c, max_gamma = select_param_RBF(X_train, y_train, metric = "auroc",k = 5, C_range = c_range, gamma_range=gamma_range)
    print(f"max_c:{max_c}, max_gamma:{max_gamma}")
    """
    #question42(X_train, y_train, X_test, y_test, metric_list)
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       helper.save_challenge_predictions to save your predicted labels
    X_challenge, y_challenge, X_heldout, feature_names = helper.get_challenge_data()
    ncount = np.sum(y_challenge == -1)
    pcount = np.sum(y_challenge == 1)
    
    wp = (ncount + pcount)/(2*pcount)
    wn = (ncount+pcount)/(2*ncount)
    clf = LogisticRegression(
            penalty="l2",
            C=1,
            solver = "liblinear",
            class_weight= {-1:wn, 1:wp},
            fit_intercept=False,
            random_state=seed
        )
    clf.fit(X_challenge,y_challenge)
    y_pred = clf.predict(X_challenge)
    confusion = metrics.confusion_matrix(y_challenge, y_pred)
    print(confusion)

    y_labels = clf.predict(X_heldout).astype(int)
    y_scores = clf.decision_function(X_heldout)

    helper.save_challenge_predictions(y_labels,y_scores, "yihanlei")


if __name__ == "__main__":
    main()
