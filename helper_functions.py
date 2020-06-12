import numpy as np
import pandas as pd

from time import time
from IPython.display import display_markdown

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from xgboost import XGBClassifier

import matplotlib as mpl
import matplotlib.pyplot as plt

# Set matplotlib color cycle
viridis_colors = plt.cm.viridis(np.linspace(0.1,0.9,6))
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=viridis_colors)

# Set matplotlib color map
mpl.rc('image', cmap='viridis')

def onehot_encode_columns(data: 'DataFrame', col_names: list) -> 'DataFrame':
    '''One hot encode one or more columns
    
    takes data frame and list of features to one hot encode. Returns dataframe
    with features onehot encoded, original featuer columns are dropped.'''
    
    for col_name in col_names:
        # make onehot encoder
        onehot_encoder = OneHotEncoder(sparse=False)

        # extract target column
        feature_array = np.array(data[col_name]).reshape(-1, 1)

        # onehot encode
        onehot_array = onehot_encoder.fit_transform(feature_array).astype('int32')

        # convert one hot encoded months to dataframe
        onehot_df = pd.DataFrame(onehot_array)
        onehot_df.columns = onehot_encoder.get_feature_names()
        onehot_df.columns = onehot_df.columns.str.replace('x0_', '')

        # concatenat onehot encoded feature back to original dataframe
        data = pd.concat([data, onehot_df], axis = 1)

        # drop column we just onehotted
        data.drop(col_name, axis=1, inplace=True)

    return data

def tune_hyperparameters(
    known_params,
    param_dist, 
    x_train, 
    y_train, 
    num_jobs, 
    search_iterations, 
    search_scoring_func
):

    # initalize XGBoost classifier
    xgb_mod = XGBClassifier(**known_params)

    # set up random search
    xgb_random_search = RandomizedSearchCV(
        xgb_mod, 
        param_distributions=param_dist,
        scoring=search_scoring_func,
        n_iter=search_iterations,
        n_jobs=num_jobs
    )

    # run and time search
    start = time()
    xgb_best_model = xgb_random_search.fit(x_train, y_train)
    print("RandomizedSearchCV took %.f min. for %d candidate"
          " parameter settings." % (((time() - start)/60), search_iterations))
    
    return xgb_best_model, xgb_random_search

# score and show confusion matrix
def print_model_score(model, x_train, y_train, x_test, y_test):
    training_score = matthews_corrcoef(model.predict(x_train), y_train)
    test_score = matthews_corrcoef(model.predict(x_test), y_test)
    
    display_markdown('**Matthews correlation coeff., training set: {}**'.format(np.round(training_score, 2)), raw=True)
    display_markdown('**Matthews correlation coeff., test set: {}**'.format(np.round(test_score, 2)), raw=True)

    
def display_confusion_matrix(model, class_names, x_test, y_test):

    raw_cm = confusion_matrix(y_test, model.predict(x_test))
    print("Raw count confusion matrix")
    print(raw_cm)
    
    normalized_cm = plot_confusion_matrix(model, x_test, y_test,
                                 display_labels=class_names,
                                 normalize='true')

    normalized_cm.ax_.set_title("Normalized confusion matrix")

    plt.show()