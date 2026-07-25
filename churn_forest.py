import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This will show all columns in dataframe.
pd.set_option('display.max_columns', None)

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier

# This will save our models once we fit them.
import pickle

# The script below is for opening this .py file on a Mac or other devices not the iPad. 
# I will call this section #Change directory# and it goes to the end of this section.
import os
# Get the absolute directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
# Change path to directory of current script. 
os.chdir(script_dir)
# End of #Change directory# section.

df = pd.read_csv('Churn_Modelling.csv')
print(df.head())

# Let's drop columns we don't need. 

churn_df = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Gender'], axis=1)

print(churn_df.head())

# Now we change Geography column to integers. 
# We will only keep Spain and Germany, because if neithe is true, then it's France!

churn_df2 = pd.get_dummies(churn_df, columns=['Geography'], prefix='Geo', drop_first=True, dtype=int)
print(churn_df2.head())

# Split the data and apply train_test_split. 
# Define your X and y variables.
# We will use 25% test data, and 75% train data.

y = churn_df2['Exited']

X = churn_df2.copy()
X = X.drop('Exited', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Now we apply Random Forest classifier, and use GridsearchCV on it. 

print(X_train.shape)
print(y_train.shape)

rf = RandomForestClassifier(random_state=0, n_jobs=1)

cv_params = {'max_depth': [2,3,4,5, None],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]}

scoring = ['accuracy', 'precision', 'recall', 'f1']
# This is where the script takes a long time. 
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='f1', n_jobs=-1)

rf_cv.fit(X_train, y_train)

print(rf_cv.best_params_)

print(rf_cv.best_score_)

# Let's create a table of scores from the decision tree. 
# Define a new function to do this.


def make_results(model_name, model_object):
    cv_results = pd.DataFrame(model_object.cv_results_)
    best_est_res = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]
    f1 = best_est_res.mean_test_f1
    accuracy = best_est_res.mean_test_accuracy
    recall = best_est_res.mean_test_recall
    precision = best_est_res.mean_test_precision
    # Now create a table of scores
    table = pd.DataFrame()
    table = pd.DataFrame({'Model': [model_name],
                          'Accuracy': [accuracy],
                          'Precision': [precision],
                          'Recall': [recall],
                          'F1': [f1]})
    return table

rf_cv_results = make_results('Random Forest CV', rf_cv)
print(rf_cv_results)

results = pd.read_csv('churn_tree_results.csv', index_col=0)
print(results)

# Let's make validation set.

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                            stratify=y_train, random_state=10)

split_index = [0 if x in X_val.index else -1 for x in X_train.index]

from sklearn.model_selection import PredefinedSplit

rfv = RandomForestClassifier(random_state=0, n_jobs=1)

cv_params = {'max_depth': [2,3,4,5, None], 
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'max_features': [2,3,4],
             'n_estimators': [75, 100, 125, 150]
             } 

scoring = ['accuracy', 'precision', 'recall', 'f1']

custom_split = PredefinedSplit(split_index)

rf_val = GridSearchCV(rfv, cv_params, scoring=scoring, cv=custom_split, refit='f1', n_jobs=-1)

rf_val.fit(X_train, y_train)

rf_val.best_params_

rf_val_results = make_results('Random Forest Validated', rf_val)
print(rf_val_results)


