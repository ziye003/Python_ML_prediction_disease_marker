# Databricks notebook source
pip install --upgrade scikit-learn

# COMMAND ----------

from __future__ import print_function
import time
import numpy as np
import pandas as pd
#from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import math
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
%matplotlib inline
import numpy as np
from scipy import stats
import math
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
%matplotlib inline
import numpy as np
from scipy import stats
from statistics import mean
from statistics import median
import sklearn.datasets
import pandas as pd
import numpy as np
# import umap
# import cupy as cp

# COMMAND ----------

# MAGIC %md # load data

# COMMAND ----------

common_hypoxia_marker=pd.read_csv('/dbfs/mnt/sapbio-client-002sap/002SAP21P009-Maltepe-lamb-hypoxia/04-DataAnalysis/analysis/lamb_pig_common_hypoxia_marker.csv')

# COMMAND ----------

# MAGIC %md # train test split

# COMMAND ----------

# load data
data = feature_input.values
X, y = feature_input.iloc[:, :-1], feature_input.iloc[:, -1]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)


features = feature_input.columns[:-1]

# names = Logistic_input_df.columns[:-2]
X_train, X_test, y_train, y_test = train_test_split(
    feature_input.drop(labels=['num_outcome'], axis=1),
    feature_input['num_outcome'],
    test_size=0.4,
    random_state=0)
X_train.shape, X_test.shape,y_train.shape, y_test.shape,X_sc.shape, X.shape,

# COMMAND ----------

# MAGIC %md # feature selection

# COMMAND ----------

# MAGIC %md ## lasso

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.linear_model import Lasso, LogisticRegression
# MAGIC from sklearn.feature_selection import SelectFromModel
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC 
# MAGIC # https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
# MAGIC from numpy import mean
# MAGIC from numpy import std
# MAGIC from numpy import absolute
# MAGIC from pandas import read_csv
# MAGIC from sklearn.model_selection import cross_val_score
# MAGIC from sklearn.model_selection import RepeatedKFold
# MAGIC from sklearn.linear_model import Lasso
# MAGIC import numpy as np
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC from sklearn.pipeline import Pipeline
# MAGIC from sklearn.model_selection import train_test_split, GridSearchCV

# COMMAND ----------

# MAGIC %md ### search for best param

# COMMAND ----------

#define model
model = LogisticRegression(penalty='l1')

#param value
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'solver':['saga','liblinear']}

# define evaluation
cv = RepeatedKFold(n_splits=2, n_repeats=30, random_state=0)

# define search
search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=cv)
# search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=cv)
# execute search
# result = search.fit(X_train, y_train)
result = search.fit(X_sc, y.astype(str))

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

# MAGIC %md ###Check prediciton

# COMMAND ----------

#Fitting the model

model = LogisticRegression(C=0.001,penalty='l1', solver='liblinear')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Model Predictions
from sklearn.metrics import mean_absolute_error,r2_score
print(r2_score(y_test,y_pred))

#Visualizing the results
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.distplot(y_test-y_pred)

# COMMAND ----------

# MAGIC %md ###Features

# COMMAND ----------

sel_l1 = SelectFromModel(LogisticRegression(C=1,penalty='l1', solver='saga'))
sel_l1.fit(X_sc, y)


f=pd.DataFrame(sel_l1.estimator_.coef_)
f=f.T
f.hist()

# COMMAND ----------

np.sum(sel_l1.get_support() == False)

# COMMAND ----------

selected_feat = X_train.columns[(sel_l1.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_l1.estimator_.coef_ == 0)))

# COMMAND ----------

#identify removed features
removed_feats = X_train.columns[(sel_l1.estimator_.coef_ == 0).ravel().tolist()]
selected_feats_logi_l1 = X_train.columns[(sel_l1.estimator_.coef_ != 0).ravel().tolist()]
selected_feats_logi_l1

# COMMAND ----------

# MAGIC %md ## Ridge

# COMMAND ----------

# MAGIC %md
# MAGIC ### Search for best params

# COMMAND ----------

#define model
model = LogisticRegression(penalty='l2')

#param value
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
             'solver':['newton-cg','lbfgs','liblinear','sag','saga']}

# define search
search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=cv)

# execute search
# result = search.fit(X_train, y_train)
result = search.fit(X_sc, y)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check prediction

# COMMAND ----------

#Fitting the model

model = LogisticRegression(C=0.001, penalty='l2', solver='liblinear')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Model Predictions
from sklearn.metrics import mean_absolute_error,r2_score
print(r2_score(y_test,y_pred))

#Visualizing the results
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.distplot(y_test-y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features

# COMMAND ----------

sel_l2 = SelectFromModel(LogisticRegression(C=0.001, penalty='l2', solver='liblinear'))
sel_l2.fit(X_sc, y)


f=pd.DataFrame(sel_l2.estimator_.coef_)
f=f.T
f.hist()

# COMMAND ----------

selected_feat = X_train.columns[(sel_l2.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_l2.estimator_.coef_ == 0)))

# COMMAND ----------

removed_feats = X_train.columns[(sel_l2.estimator_.coef_ == 0).ravel().tolist()]
selected_feats_logi_l2 = selected_feat
selected_feats_logi_l2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
# import shap
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import RandomForestClassifier as rfc
# from sklearn.grid_search import GridSearchCV

# COMMAND ----------

# MAGIC %md
# MAGIC ### search for best param

# COMMAND ----------

# define evaluation
cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=0)

# COMMAND ----------

rfc = RandomForestClassifier(n_jobs = -1, max_features = 'auto',oob_score=False, random_state=0) 



# rfbase = rfc(n_jobs = 3, max_features = 'auto', n_estimators = 100, bootstrap=False)

params = {
    'n_estimators': [10,20,50,100,200,500],
    'max_features': [0.3,.5,.7],
    'bootstrap': [False, True],
    'max_depth':[3,6]
}

rf_fit = GridSearchCV(estimator=rfc, param_grid=params , scoring = 'roc_auc')
# neg_mean_absolute_error
# search = GridSearchCV(estimator=rfc, param_grid=params,scoring='roc_auc', n_jobs=-1,  cv= cv)

result = rf_fit.fit(X_sc, y.astype(str))

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check prediction

# COMMAND ----------

#Fitting the model
model = RandomForestClassifier(n_estimators=50,max_features=0.5,bootstrap=True,n_jobs = -1, max_depth = 3,oob_score=False, random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Model Predictions
from sklearn.metrics import mean_absolute_error,r2_score
print(r2_score(y_test,y_pred))

#Visualizing the results
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.distplot(y_test-y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Features

# COMMAND ----------

model.fit(X_sc,y)
f=pd.DataFrame(model.feature_importances_)
# f=f.T
f
f.hist()

# COMMAND ----------

sorted_idx = model.feature_importances_.argsort()
feature_df=pd.DataFrame(model.feature_importances_[sorted_idx],features[sorted_idx],columns=['Feature Importance'])

# selected_feat = X_train.columns[(sel_.get_support())]
# print('total features: {}'.format((X_train.shape[1])))
# print('selected features: {}'.format(len(selected_feat)))
# print('features with coefficients shrank to zero: {}'.format(
#       np.sum(sel_.estimator_.coef_ == 0)))

# feature_df=feature_df[feature_df['Alpha = 0.010000']!=0]
print(feature_df.shape)
feature_df
selected_feats_RF=feature_df[feature_df['Feature Importance']!=0].index
len(selected_feats_RF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary of features

# COMMAND ----------

!pip install matplotlib_venn

import matplotlib.pyplot as plt
import matplotlib_venn as venn
from matplotlib_venn import venn2
from matplotlib_venn import venn3

# COMMAND ----------

print(len(selected_feats_logi_l1))
print(len(selected_feats_logi_l2))
print(len(selected_feats_RF))

# COMMAND ----------

venn2([set(selected_feats_logi_l1), set(selected_feats_logi_l2)],('Logstic_L1','Logstic_L2'),set_colors=("red",'orange'), alpha = 1)

# COMMAND ----------

venn3([set(selected_feats_logi_l1), set(selected_feats_logi_l2),set(selected_feats_RF)],('Logstic_L1','Logstic_L2','Random_Forest'),set_colors=("blue", "red",'orange'), alpha = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Multivariate analysis

# COMMAND ----------

# https://www.kaggle.com/phamvanvung/partial-least-squares-regression-in-python
# https://nirpyresearch.com/pls-discriminant-analysis-binary-classification-python/
# https://towardsdatascience.com/feature-selection-in-machine-learning-using-lasso-regression-7809c7c2771a
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

# COMMAND ----------


print(feature_df.head())
# abs(feature_df).nlargest(3,1)
feature_df.columns
Top3RF=abs(feature_df).nlargest(3,'Feature Importance').index
print(abs(feature_df).nlargest(3,'Feature Importance'))
Top3RF

# COMMAND ----------

# # selected_feat = X_train.columns[(sel_l2.get_support())]
# print('total features: {}'.format((X_train.shape[1])))
# print('selected features: {}'.format(len(selected_feat)))
# print('features with coefficients shrank to zero: {}'.format(
#       np.sum(sel_l2.estimator_.coef_ == 0)))
# print(sel_l2.estimator_.coef_.shape)
# print(feature_input.columns.shape)
Top3L2_df=pd.DataFrame(sel_l1.estimator_.coef_,columns=feature_input.columns[:-1]).T
print(Top3L2_df.head())
print(abs(Top3L2_df).nlargest(3,0))
Top3L2=abs(Top3L2_df).nlargest(3,0).index
print(Top3L2)

# COMMAND ----------

# Read data into a numpy array and apply simple smoothing
# features=selected_feats_RF
# features=selected_feats_logi_l2
# features=Top3L2
features=Top3RF

featureColumns=features.to_list()
print(len(featureColumns))
featureColumns.append('num_outcome')

# features=feature_input.columns
# featureColumns=features.to_list()



predict_input=feature_input[featureColumns]
# load data
data = predict_input.values
X, y = predict_input.iloc[:, :-1], predict_input.iloc[:, -1]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)


features = predict_input.columns[:-1]

# names = Logistic_input_df.columns[:-2]
X_train, X_test, y_train, y_test = train_test_split(
    predict_input.drop(labels=['num_outcome'], axis=1),
    predict_input['num_outcome'],
    test_size=0.4,
    random_state=0)
X_train.shape, X_test.shape,y_train.shape, y_test.shape,X_sc.shape, X.shape
# # X_binary=Logistic_input_df.drop(labels=['Asphyxia_x',  'clinical_outcome'	,'num_outcome'], axis=1)
# # num_df=Logistic_input_df.drop(labels=['Asphyxia_x',  'clinical_outcome'	,'num_outcome'], axis=1) # results from biological replicated
# scaler = StandardScaler()
# num_df=Logistic_input_df.drop(labels=['Asphyxia_x',  'clinical_outcome'	,'num_outcome'], axis=1) 
# # PLSDA_input=num_df[selected_feats_Linear_L1]
# PLSDA_input=num_df[selected_feats_logi_l1]
# # PLSDA_input=num_df[top1metabolites] # results from best logistic outcome 

# # PLSDA_input=num_df[comoon_L1_RF_list]
# X_binary = PLSDA_input

# X_sc = scaler.fit_transform(X_binary)

# # Read categorical variables
# y_binary = Logistic_input_df['num_outcome'].values
# X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=0)

# # Map variables to 0 and 1
# # y_binary = (y_binary == 1).astype('uint8')
# X_binary.shape,y_binary.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ###LDA 

# COMMAND ----------

from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# COMMAND ----------

# MAGIC %md #### parameter optimization

# COMMAND ----------

# create the lda model
model = LinearDiscriminantAnalysis(solver='lsqr')
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
params = {
    'shrinkage': [0, 1, 0.01]
}

search = GridSearchCV(estimator=model, param_grid=params , scoring = 'roc_auc',n_jobs=-1, cv=cv)
# neg_mean_absolute_error,accuracy
# search = GridSearchCV(estimator=rfc, param_grid=params,scoring='roc_auc', n_jobs=-1,  cv= cv)
# results = search.fit(X, y)
result = search.fit(X_train, y_train.astype(str))
print(X_train.shape)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

model = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.0)
# evaluate model
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('roc_auc: %.3f (%.3f)' % (mean(scores), std(scores)))

# COMMAND ----------

model.fit(X_train, y_train)
lda_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
lda_disp

# COMMAND ----------

# MAGIC %md ### xgb

# COMMAND ----------

!pip install xgboost

# COMMAND ----------

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# COMMAND ----------

# MAGIC %md #### parameter optimisation

# COMMAND ----------

# create the lda model
model = XGBClassifier()
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
params = {
    'max_depth': range(1, 10, 5),
    'n_estimators': range(30,100,10),
    'learning_rate': [0.1, 0.01, 0.05]
}

search = GridSearchCV(estimator=model, param_grid=params , scoring = 'roc_auc',n_jobs=-1, cv=cv)
# neg_mean_absolute_error,accuracy
# search = GridSearchCV(estimator=rfc, param_grid=params,scoring='roc_auc', n_jobs=-1,  cv= cv)
# results = search.fit(X, y)
result = search.fit(X_train, y_train.astype(str))
print(X_train.shape)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

# evaluate model
model= XGBClassifier(max_depth=1,n_estimators=30,learning_rate=0.01)
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
# scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=1, n_repeats=1, random_state=42), n_jobs=-1)

# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('roc_auc: %.3f (%.3f)' % (mean(scores), std(scores)))

# COMMAND ----------

model.fit(X_train, y_train)
xgb_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
xbg_disp

# COMMAND ----------

# MAGIC %md ## PLS DA

# COMMAND ----------

# https://www.kaggle.com/phamvanvung/partial-least-squares-regression-in-python
# https://nirpyresearch.com/pls-discriminant-analysis-binary-classification-python/
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md #### parameter optimisation

# COMMAND ----------

 
model = PLSRegression()
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
params={'n_components':[1,2,3,4,5,6]}

# search = GridSearchCV(estimator=model, param_grid=params , scoring = 'accuracy',n_jobs=-1, cv=cv)
search = GridSearchCV(estimator=model, param_grid=params , scoring = 'roc_auc',n_jobs=-1, cv=cv)

# neg_mean_absolute_error,accuracy
# search = GridSearchCV(estimator=rfc, param_grid=params,scoring='roc_auc', n_jobs=-1,  cv= cv)
# results = search.fit(X, y)
result = search.fit(X_train, y_train.astype(str))

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------


# Define the PLS regression object
X_binary=X_train
y_binary=y_train.astype('float').to_list()
# Define the PLS regression object
pls_binary =PLSRegression(n_components=4)
# Fit and transform the data
X_pls = pls_binary.fit_transform(X_binary, y_binary)[0]

# COMMAND ----------

# Scatter plot
unique = list(set(y_binary))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
 
# with plt.style.context(('ggplot')):
#     plt.figure(figsize=(12,10))
#     for i, u in enumerate(unique):
#         col = np.expand_dims(np.array(colors[i]), axis=0)

#         xi = [X_pls[j,0] for j in range(len(X_pls[:,0])) if y_binary[j] == u]
#         yi = [X_pls[j,1] for j in range(len(X_pls[:,1])) if y_binary[j] == u]
#         plt.scatter(xi, yi, c=col, s=100, edgecolors='k',label=str(u))
 
#     plt.xlabel('Latent Variable 1')
#     plt.ylabel('Latent Variable 2')
#     plt.legend(labplot,loc='lower left')
#     plt.title('PLS cross-decomposition')
#     plt.show()

# scores.index=features

with plt.style.context(('ggplot')):
    plt.figure(figsize=(12,10))
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [X_pls[j,0] for j in range(len(X_pls[:,0])) if y_binary[j] == u]
        yi = [X_pls[j,1] for j in range(len(X_pls[:,1])) if y_binary[j] == u]
        plt.rcParams['axes.facecolor'] = '1.0'
#         fig = plt.figure()
#         ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
#         col=['#1f77b4', '#ff7f0e']
        plt.scatter(xi, yi, c=col, s=100,label=str(u))
 
    plt.xlabel('Latent Variable 1',fontsize=40)
    plt.ylabel('Latent Variable 2',fontsize=40)
#     plt.legend(labplot,loc='lower left',fontsize=20)
    plt.legend(labplot,loc=0,fontsize=20)
    plt.title('PLS cross-decomposition',fontsize=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

# COMMAND ----------

myplsda=PLSRegression(n_components=4).fit(X_train, y_train)

# COMMAND ----------

mypred= myplsda.predict(X_train)

# COMMAND ----------


unique = list(set(y_train))
colors
col = np.expand_dims(np.array(colors[i]), axis=0)
col

# COMMAND ----------


# Define the labels for the plot legend
labplot = ["Unaffected", "cerebral palsy-like & spastic diparesis"]
# Scatter plot

# Define the PLS regression object
pls_binary =PLSRegression(n_components=2)
# Fit and transform the data
X_pls = pls_binary.fit_transform(X_train, y_train.astype('uint8'))[0]
# Scatter plot
unique = list(set(y_binary))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
 
with plt.style.context(('ggplot')):
    plt.figure(figsize=(12,10))
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [X_pls[j,0] for j in range(len(X_pls[:,0])) if y_binary[j] == u]
        yi = [X_pls[j,1] for j in range(len(X_pls[:,1])) if y_binary[j] == u]
        plt.scatter(xi, yi, c=col, s=100, edgecolors='k',label=str(u))
 
    plt.xlabel('Latent Variable 1')
    plt.ylabel('Latent Variable 2')
    plt.legend(labplot,loc='lower left')
    plt.title('PLS cross-decomposition')
    plt.show()

# COMMAND ----------

# Test-train split
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=0)
# Define the PLS object
pls_binary = PLSRegression(n_components=2)
# Fit the training set
pls_binary.fit(X_train, y_train)
 
# Predictions: these won't generally be integer numbers
y_pred = pls_binary.predict(X_test)[:,0]
# "Force" binary prediction by thresholding
binary_prediction = (pls_binary.predict(X_test)[:,0] >= 0.5).astype('uint8')
print(binary_prediction, y_test)

# COMMAND ----------


def pls_da(X_train,y_train, X_test):
    
    # Define the PLS object for binary classification
    plsda = PLSRegression(n_components=2)
    
    # Fit the training set
    plsda.fit(X_train, y_train)
    
    # Binary prediction on the test set, done with thresholding
    binary_prediction = (plsda.predict(X_test)[:,0] >= 0.5).astype('uint8')
    
    return binary_prediction

# COMMAND ----------


accuracy = list()
cval = KFold(n_splits=2, shuffle=True, random_state=0)
for train, test in cval.split(X_binary):
  
  y_pred = pls_da(X_binary.iloc[train,:], y_binary[train], X_binary.iloc[test,:])

  accuracy.append(accuracy_score(y_binary[test], y_pred))
 


# COMMAND ----------

print("Average accuracy on 2 splits: ", np.array(accuracy).mean())

# COMMAND ----------

# MAGIC %md
# MAGIC # Random forest

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict, train_test_split


# COMMAND ----------

# MAGIC %md #### parameter optimization

# COMMAND ----------


model = RandomForestClassifier()
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)
rfc = RandomForestClassifier(n_jobs = -1, max_features = 'auto',oob_score=False, random_state=0) 



# rfbase = rfc(n_jobs = 3, max_features = 'auto', n_estimators = 100, bootstrap=False)

params = {
    'n_estimators': [10,20,50,100,200,500],
    'max_features': [0.3,.5,.7],
    'bootstrap': [False, True],
    'max_depth':[3,6]
}

rf_fit = GridSearchCV(estimator=rfc, param_grid=params , scoring = 'roc_auc')
# neg_mean_absolute_error
# search = GridSearchCV(estimator=rfc, param_grid=params,scoring='roc_auc', n_jobs=-1,  cv= cv)

result = rf_fit.fit(X_train, y_train.astype(str))

# summarize result

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

model = RandomForestClassifier(n_estimators=20,max_features=0.3,bootstrap=False,n_jobs = -1, max_depth = 6,oob_score=False, random_state=0)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# COMMAND ----------

scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
print('AUCROC: %.3f (%.3f)' % (mean(scores), std(scores)))
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)

print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# COMMAND ----------


# evaluate model
scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
model.fit(X_train, y_train)
rf_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
rf_disp

# COMMAND ----------

def rf_pred(X_train,y_train, X_test):
    
    # Define the PLS object for binary classification
    rf = RandomForestClassifier(n_estimators=20,max_features=0.3,bootstrap=True,n_jobs = -1, max_depth = 3,oob_score=False, random_state=0)
    
    # Fit the training set
    rf.fit(X_train, y_train)
    
    # Binary prediction on the test set, done with thresholding
    binary_prediction = rf.predict(X_test)
    
    return binary_prediction
  


# COMMAND ----------

accuracy = list()
cval = KFold(n_splits=2, shuffle=True, random_state=0)
for train, test in cval.split(X_binary):
  
  y_pred = rf_pred(X_binary.iloc[train,:], y_binary[train], X_binary.iloc[test,:])

  accuracy.append(accuracy_score(y_binary[test], y_pred))
print("Average accuracy on 50 splits: ", np.array(accuracy).mean())

# COMMAND ----------

rf = RandomForestClassifier(n_estimators=20,max_features=0.3,bootstrap=True,n_jobs = -1, max_depth = 3,oob_score=False, random_state=0)
rf.fit(X_train, y_train)
rfc_disp = RocCurveDisplay.from_estimator(rf, X_test, y_test)
rfc_disp

# COMMAND ----------

# MAGIC %md # Volcano plot

# COMMAND ----------

severity_df=pd.read_csv('/dbfs/mnt/sapbio-client-002sap/002SAP21P009-Maltepe-lamb-hypoxia/04-DataAnalysis/analysis/EOA_slope_df.csv')
from scipy.stats import t
t_stat = severity_df.t
dof = 30
t_stat=2*(1 - t.cdf(abs(t_stat), dof))
severity_df['pvalue']=t_stat

# COMMAND ----------

hypoxia_df_t=pd.read_csv('/dbfs/mnt/sapbio-client-002sap/002SAP21P009-Maltepe-lamb-hypoxia/04-DataAnalysis/analysis/pairedt_EOA_lamb_t.csv')

hypoxia_df=hypoxia_df_t.T
# maltepe_df.index=maltepe_df.loc[:,'Unnamed: 0']
hypoxia_df.columns=hypoxia_df.loc['Unnamed: 0',:]
hypoxia_df=hypoxia_df.iloc[1:,:]


# COMMAND ----------

def scatter_plot(scatter_df,mmolecules,CoeThreshold,logPThreshold):

    scatter_df['Molecule'] = 'Not Significant'
    scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope>=CoeThreshold)] ='Significant Up'
    scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope<=-CoeThreshold)] ='Significant Down'
    print(scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope>=CoeThreshold)].shape)
    print(scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope<=-CoeThreshold)].shape)
    plt.figure(figsize=(15,15))
    hue_order=['Not Significant','Significant Down','Significant Up']
    
    ax=sns.scatterplot(data=scatter_df, x="HypoxiaTime_slope", y="log_P",hue='Molecule',
                       palette=['lightgrey','#44DAED','#ed6c21'],
                       hue_order=hue_order,
                       s=150)
    ax.set_xlim([-20, 20])
#     plt.text(x=scatter_df.loc[mmolecules,'HypoxiaTime_slope'],y=scatter_df.loc[mmolecules,'log_P'],s=mmolecules,fontsize=30)
#     i=mmolecules
#     plt.text(x=scatter_df.loc[mmolecules,'HypoxiaTime_slope'],y=scatter_df.loc[mmolecules,'log_P'],s=mmolecules,fontsize=9)

#     ax.xticks(np.arange(min(x), max(x), (max(x)-min(x))/4))
                      
    ax.set_yticklabels(ax.get_yticks(), size = 30)
    
#     xticklbls = []
#     for xtick in ax.get_xticks():
#       xticklbls.append(f'{xtick:.2e}')
#     ax.set_xticklabels(xticklbls, size = 26)
#     plt.xlim([-50, 50])
    ax.set_xticklabels(ax.get_xticks(), size = 26)
    

    plt.legend(loc='upper right', borderaxespad=0)
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='30') # for legend title
    

    ax.set_xlabel('Effect Size',fontsize=50)
    ax.set_ylabel('-Log10 (p_value)',fontsize=50)
      
    
#     if molecules != 'no':
    for i in mmolecules:
      plt.text(x=scatter_df.loc[i,'HypoxiaTime_slope'],y=scatter_df.loc[i,'log_P'],s=i,fontsize=20)
#     print('up'+scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope>=CoeThreshold)].shape)
#     print(scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope<=-CoeThreshold)].shape)
def scatter_plot2(scatter_df,molecules,CoeThreshold,logPThreshold):

    scatter_df['Molecule'] = 'Not Significant'
    scatter_df['Molecule'][(scatter_df.log_P>=logPThreshold)&(scatter_df.HypoxiaTime_slope>=CoeThreshold)] ='Significant Up'
    scatter_df['Molecule'][(scatter_df.log_P>=logPThreshold)&(scatter_df.HypoxiaTime_slope<=-CoeThreshold)] ='Significant Down'
    scatter_df.loc[molecules,'Molecule'] ='Markers'
    print(scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope>=CoeThreshold)].shape)
    print(scatter_df['Molecule'][(scatter_df.log_P>logPThreshold)&(scatter_df.HypoxiaTime_slope<=-CoeThreshold)].shape)
    plt.figure(figsize=(15,15))
#     hue_order=['Markers','Not Significant','Significant Down','Significant Up']
#     ax=sns.scatterplot(data=scatter_df, x="HypoxiaTime_slope", y="log_P",hue='Molecule',
#                        palette=['purple','lightgrey','#44DAED','#ed6c21'],
#                        hue_order=hue_order,
#                        size='Molecule',
#                        sizes=[200,20,20,20], size_order=hue_order)

    hue_order=['Not Significant','Significant Down','Significant Up','Markers']
    ax=sns.scatterplot(data=scatter_df, x="HypoxiaTime_slope", y="log_P",hue='Molecule',
                       palette=['lightgrey','#44DAED','#ed6c21','purple'],
                       hue_order=hue_order,
                       size='Molecule',
                       alpha=0.9,
                       sizes=[20,20,20,200], size_order=hue_order)

    ax.set_xlim([-20, 20])
    ax.set_yticklabels(ax.get_yticks(), size = 30)
    
#     xticklbls = []
#     for xtick in ax.get_xticks():
#       xticklbls.append(f'{xtick:.2e}')
    
#     print(xticklbls)
    ax.set_xticklabels(ax.get_xticks(), size = 30)
#     plt.xlim(-10, 10)
#     plt.xlim([-10, 10])
#     plt.legend(loc='lower left', borderaxespad=0)
    plt.legend(loc='best', borderaxespad=0)
    plt.setp(ax.get_legend().get_texts(), fontsize='30') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='30') # for legend title
    
#     plt.xlim(-5, 5)
#     ax.set_xlabel('Effect Size',fontsize=50)
#     ax.set_xlabel('Log2 (Fold Change in Median)',fontsize=50)
    ax.set_xlabel('Effect Size',fontsize=50)
    ax.set_ylabel('-Log10 (p_value)',fontsize=50)

# COMMAND ----------

selected_feats_logi_l2.to_list()

# COMMAND ----------

# maltepe_df_T['pt_test_pvalue_EOA'].fillna(0.1)
# severity_df=maltepe_df
severity_df['log_P']=-np.log10(severity_df['pvalue'])
severity_df['HypoxiaTime_slope']=severity_df['slope']
severity_df['molecule_name']=severity_df['metabolite']
severity_df.index=severity_df['metabolite']
severity_df.head()

# COMMAND ----------

hypoxia_df.head()

# listr2=selected_feats_logi_l2.drop('rLC_neg_mtb_3232944')
# listr2

# COMMAND ----------

# hypoxia_df.head()
hypoxia_df['log_P']=-np.log10(hypoxia_df['pt_test_pvalue_EOA'].astype(float))
hypoxia_df['HypoxiaTime_slope']=np.log2(hypoxia_df['FC_median'].astype(float))
hypoxia_df['molecule_name']=hypoxia_df.index
# hypoxia_df.index=hypoxia_df['metabolite']

# COMMAND ----------

Top3RF

# COMMAND ----------

hypoxia_df[hypoxia_df.index=='rLC_pos_mtb_1319082']

# COMMAND ----------

scatter_plot2(hypoxia_df,['rLC_neg_mtb_5920244', 'rLC_pos_mtb_2264309', 'rLC_pos_mtb_1319082'],CoeThreshold=0,logPThreshold=6) 

# COMMAND ----------

scatter_plot(hypoxia_df,['rLC_neg_mtb_5920244', 'rLC_pos_mtb_2264309', 'rLC_pos_mtb_1319082'],CoeThreshold=0,logPThreshold=6) 

# COMMAND ----------

scatter_plot(severity_df,Top3RF,CoeThreshold=0,logPThreshold=2) 

# COMMAND ----------

# MAGIC %md # linear combination

# COMMAND ----------

# MAGIC %md ## PLS linear combination input

# COMMAND ----------

# features=selected_feats_RF
features=selected_feats_logi_l2
featureColumns=features.to_list()
featureColumns.append('num_outcome')

predict_input=feature_input[featureColumns]
# load data
data = predict_input.values
X, y = predict_input.iloc[:, :-1], predict_input.iloc[:, -1]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)


features = predict_input.columns[:-1]

# names = Logistic_input_df.columns[:-2]
X_train, X_test, y_train, y_test = train_test_split(
    predict_input.drop(labels=['num_outcome'], axis=1),
    predict_input['num_outcome'],
    test_size=0.3,
    random_state=0)
X_train.shape, X_test.shape,y_train.shape, y_test.shape,X_sc.shape, X.shape

# COMMAND ----------

pls_binary =PLSRegression(n_components=2)
# Fit and transform the data
X_pls = pls_binary.fit_transform(X_train, y_train)[0]
# print(X_pls.shape)
# X_pls

# COMMAND ----------

pls_binary.x_weights_.shape
x1=np.dot(X_sc,pls_binary.x_weights_[:,0])
x2=np.dot(X_sc,pls_binary.x_weights_[:,1])

# COMMAND ----------

PLS_X=pd.DataFrame([x1,x2])
PLS_X

# COMMAND ----------

linear_factor=PLS_X.T
# 

# COMMAND ----------

# MAGIC %md ##Lasso combination input

# COMMAND ----------

print(sel_l2.estimator_.coef_.shape)
feature_input.iloc[:, :-1].shape

# COMMAND ----------

linear_factor=np.dot(feature_input.iloc[:, :-1].values, sel_l1.estimator_.coef_.T)

# COMMAND ----------

linear_factor.shape

# COMMAND ----------

# MAGIC %md ## lasso

# COMMAND ----------

X, y = linear_factor, feature_input.iloc[:, -1]

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_sc,
    y,
    test_size=0.4,
    random_state=0)
X_train.shape, X_test.shape,y_train.shape, y_test.shape,X_sc.shape, X.shape

#define model
model = LogisticRegression(solver='liblinear')

#param value
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

# define evaluation
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)

# define search
search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=cv)

# execute search
# result = search.fit(X_train, y_train)
result = search.fit(X_train, y_train)

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

model = LogisticRegression(solver='liblinear',C=100).fit(X_train, y_train)
pred=model.predict(X_sc)

linear_factor_df=pd.DataFrame(X_sc)
# linear_factor_df=pd.DataFrame(linear_factor)
pred_df=pd.DataFrame(pred)
outcome_df=pd.concat([linear_factor_df,pred_df],axis=1)       
outcome_df
linear_factor_df.hist()

# COMMAND ----------

# evaluate model
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('ROC_AUC: %.3f (%.3f)' % (mean(scores), std(scores)))

# COMMAND ----------

model.fit(X_train, y_train)
logit_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
logit_disp

# COMMAND ----------

# MAGIC %md ## lda

# COMMAND ----------

# create the lda model
model = LinearDiscriminantAnalysis(solver='lsqr')
cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
params = {
    'shrinkage': [0, 1, 0.01]
}

search = GridSearchCV(estimator=model, param_grid=params , scoring = 'roc_auc',n_jobs=-1, cv=cv)
# neg_mean_absolute_error,accuracy
# search = GridSearchCV(estimator=rfc, param_grid=params,scoring='roc_auc', n_jobs=-1,  cv= cv)
# results = search.fit(X, y)
result = search.fit(X_train, y_train.astype(str))

# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

# COMMAND ----------

model = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0)
# evaluate model
scores = cross_val_score(model, X_test, y_test, scoring='accuracy', cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
scores = cross_val_score(model, X_test, y_test, scoring='roc_auc', cv=RepeatedKFold(n_splits=2, n_repeats=2, random_state=42), n_jobs=-1)
# summarize result
print('ROC_AUC: %.3f (%.3f)' % (mean(scores), std(scores)))

# COMMAND ----------

model.fit(X_train, y_train)
lda_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)
lda_disp

# COMMAND ----------


