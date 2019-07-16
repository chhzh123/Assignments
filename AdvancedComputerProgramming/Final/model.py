import sys
import gc
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import norm, skew, kurtosis  # for some statistics
import IPython
from IPython import display
import sklearn
import random
import time
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl

# %matplotlib inline
# ignore warnings
import warnings

warnings.filterwarnings('ignore')
print('-' * 25)

input_path = 'data/'
train_data = pd.read_csv(input_path + 'happiness_train_complete.csv', sep=',', encoding='latin-1')
test_data = pd.read_csv(input_path + 'happiness_test_complete.csv', sep=',', encoding='latin-1')
submit_example = pd.read_csv(input_path + 'happiness_submit.csv', sep=',', encoding='latin-1')

print('train shape:', train_data.shape)
print('test shape:', test_data.shape)
print('sample shape:', submit_example.shape)

train_data = train_data[train_data["happiness"] != -8].reset_index(drop=True)  # drop=True
print('train shape:', train_data.shape)
train_data_copy = train_data.copy()
target_col = "happiness"
target = train_data_copy[target_col]  # .apply(lambda x:np.log1p(x))
del train_data_copy[target_col]

train_shape = train_data.shape[0]
data = pd.concat([train_data_copy, test_data], axis=0, ignore_index=True)
data.head()


# make feature
def getres1(row):
    return len([x for x in row.values if type(x) == int and x < 0])


def getres2(row):
    return len([x for x in row.values if type(x) == int and x == -8])


def getres3(row):
    return len([x for x in row.values if type(x) == int and x == -1])


def getres4(row):
    return len([x for x in row.values if type(x) == int and x == -2])


def getres5(row):
    return len([x for x in row.values if type(x) == int and x == -3])


#
data['neg1'] = data[data.columns].apply(lambda row: getres1(row), axis=1)
data.loc[data['neg1'] > 20, 'neg1'] = 20  #

data['neg2'] = data[data.columns].apply(lambda row: getres2(row), axis=1)
data['neg3'] = data[data.columns].apply(lambda row: getres3(row), axis=1)
data['neg4'] = data[data.columns].apply(lambda row: getres4(row), axis=1)
data['neg5'] = data[data.columns].apply(lambda row: getres5(row), axis=1)

data.loc[data['health_problem'] < 0, 'health_problem'] = 0

data.loc[data['religion'] < 0, 'religion'] = 1
data.loc[data['religion_freq'] < 0, 'religion_freq'] = 1
data.loc[data['edu'] < 0, 'edu'] = 0
data.loc[data['edu_status'] < 0, 'edu_status'] = 0
data.loc[data['income'] < 0, 'income'] = 0
data.loc[data['s_income'] < 0, 's_income'] = 0
#
data.loc[(data['weight_jin'] <= 80) & (data['height_cm'] >= 160), 'weight_jin'] = data['weight_jin'] * 2
data.loc[data['weight_jin'] <= 60, 'weight_jin'] = data['weight_jin'] * 2
data.loc[data['family_income'] <= 0, 'family_income'] = 0
data.loc[data['inc_exp'] <= 0, 'inc_exp'] = 0

data.loc[data['equity'] < 0, 'equity'] = 0
data.loc[data['social_neighbor'] < 0, 'social_neighbor'] = 0

data.loc[data['class_10_after'] < 0, 'class_10_after'] = 0
data.loc[data['class_10_before'] < 0, 'class_10_before'] = 0
data.loc[data['class'] < 0, 'class'] = 0
data.loc[data['class_14'] < 0, 'class_14'] = 0
data.loc[data['family_m'] < 0, 'family_m'] = 1

data.loc[data['health'] < 0, 'health'] = 0
data.loc[data['health_problem'] < 0, 'health_problem'] = 0

data.loc[data['neighbor_familiarity'] < 0, 'neighbor_familiarity'] = 0

data.loc[data['inc_ability'] < 0, 'inc_ability'] = np.nan

data.loc[data['status_peer'] < 0, 'status_peer'] = 2
data.loc[data['status_3_before'] < 0, 'status_3_before'] = 2
#
data.loc[data['edu_yr'] < 0, 'edu_yr'] = 0
data['survey_time'] = pd.to_datetime(data['survey_time'])
data.loc[data['marital_1st'] < 0, 'marital_1st'] = np.nan
data.loc[data['marital_now'] < 0, 'marital_now'] = np.nan

data.loc[data['join_party'] < 0, 'join_party'] = np.nan

for i in range(1, 12 + 1):
    data.loc[data['leisure_' + str(i)] < 0, 'leisure_' + str(i)] = 6
for i in range(1, 9 + 1):
    data.loc[data['public_service_' + str(i)] < 0, 'public_service_' + str(i)] = data[
        'public_service_' + str(i)].dropna().mode().values
for i in range(1, 13 + 1):
    data.loc[data['trust_' + str(i)] < 0, 'trust_' + str(i)] = data['trust_' + str(i)].dropna().mode().values

data['edubir'] = data['edu_yr'] - data['birth']
#
data['survey_age'] = 2015 - data['birth']
#
data['survey_edu_age'] = 2015 - data['edu_yr']

#
data['survey_month'] = data['survey_time'].dt.month
data['survey_hour'] = data['survey_time'].dt.hour

#
data['marital_1stbir'] = data['marital_1st'] - data['birth']
#
data['marital_nowtbir'] = data['marital_now'] - data['birth']
#
data['mar'] = data['marital_nowtbir'] - data['marital_1stbir']
#
data['marital_sbir'] = data['marital_now'] - data['s_birth']
#
# data['age_'] = data['marital_nowtbir'] - data['marital_sbir']

#
data['income/s_income'] = data['income'] / (data['s_income'] + 1)
data['income+s_income'] = data['income'] + (data['s_income'] + 1)
data['income/family_income'] = data['income'] / (data['family_income'] + 1)
data['all_income/family_income'] = (data['income'] + data['s_income']) / (data['family_income'] + 1)
data['income/inc_exp'] = data['income'] / (data['inc_exp'] + 1)
data['family_income/m'] = data['family_income'] / (data['family_m'] + 0.01)

# /
data['income/floor_area'] = data['income'] / (data['floor_area'] + 0.01)
data['all_income/floor_area'] = (data['income'] + data['s_income']) / (data['floor_area'] + 0.01)
data['family_income/floor_area'] = data['family_income'] / (data['floor_area'] + 0.01)

data['income/m'] = data['floor_area'] / (data['family_m'] + 0.01)

# class
data['class_10_diff'] = (data['class_10_after'] - data['class_10_before'])
data['class_diff'] = data['class'] - data['class_10_before']
data['class_14_diff'] = data['class'] - data['class_14']

# province mean
data['province_income_mean'] = data.groupby(['province'])['income'].transform('mean').values
data['province_family_income_mean'] = data.groupby(['province'])['family_income'].transform('mean').values
data['province_equity_mean'] = data.groupby(['province'])['equity'].transform('mean').values
data['province_depression_mean'] = data.groupby(['province'])['depression'].transform('mean').values
data['province_floor_area_mean'] = data.groupby(['province'])['floor_area'].transform('mean').values

# city   mean
data['city_income_mean'] = data.groupby(['city'])['income'].transform('mean').values
data['city_family_income_mean'] = data.groupby(['city'])['family_income'].transform('mean').values
data['city_equity_mean'] = data.groupby(['city'])['equity'].transform('mean').values
data['city_depression_mean'] = data.groupby(['city'])['depression'].transform('mean').values
data['city_floor_area_mean'] = data.groupby(['city'])['floor_area'].transform('mean').values

# county  mean
data['county_income_mean'] = data.groupby(['county'])['income'].transform('mean').values
data['county_family_income_mean'] = data.groupby(['county'])['family_income'].transform('mean').values
data['county_equity_mean'] = data.groupby(['county'])['equity'].transform('mean').values
data['county_depression_mean'] = data.groupby(['county'])['depression'].transform('mean').values
data['county_floor_area_mean'] = data.groupby(['county'])['floor_area'].transform('mean').values

# ratio
data['income/county'] = data['income'] / (data['county_income_mean'] + 1)
data['family_income/county'] = data['family_income'] / (data['county_family_income_mean'] + 1)
data['equity/county'] = data['equity'] / (data['county_equity_mean'] + 1)
data['depression/county'] = data['depression'] / (data['county_depression_mean'] + 1)
data['floor_area/county'] = data['floor_area'] / (data['county_floor_area_mean'] + 1)

# age   mean
data['age_income_mean'] = data.groupby(['survey_age'])['income'].transform('mean').values
data['age_family_income_mean'] = data.groupby(['survey_age'])['family_income'].transform('mean').values
data['age_equity_mean'] = data.groupby(['survey_age'])['equity'].transform('mean').values
data['age_depression_mean'] = data.groupby(['survey_age'])['depression'].transform('mean').values
data['age_floor_area_mean'] = data.groupby(['survey_age'])['floor_area'].transform('mean').values
data['age_health_mean'] = data.groupby(['survey_age', 'gender'])['health'].transform('mean').values
data['age_edu_mean'] = data.groupby(['survey_age', 'gender'])['edu'].transform('mean').values

# age/gender   mean
data['age_income_mean'] = data.groupby(['survey_age', 'gender'])['income'].transform('mean').values
data['age_family_income_mean'] = data.groupby(['survey_age', 'gender'])['family_income'].transform('mean').values
data['age_equity_mean'] = data.groupby(['survey_age', 'gender'])['equity'].transform('mean').values
data['age_depression_mean'] = data.groupby(['survey_age', 'gender'])['depression'].transform('mean').values
data['age_floor_area_mean'] = data.groupby(['survey_age', 'gender'])['floor_area'].transform('mean').values
# data['age_BMI_mean'] = data.groupby(['survey_age','gender'])['BMI'].transform('mean').values
data['age_gender_health_mean'] = data.groupby(['survey_age', 'gender'])['health'].transform('mean').values

# class   mean
data['city_class_income_mean'] = data.groupby(['class'])['income'].transform('mean').values
data['city_class_family_income_mean'] = data.groupby(['class'])['family_income'].transform('mean').values
data['city_class_equity_mean'] = data.groupby(['class'])['equity'].transform('mean').values
data['city_class_depression_mean'] = data.groupby(['class'])['depression'].transform('mean').values
data['city_class_floor_area_mean'] = data.groupby(['class'])['floor_area'].transform('mean').values

#
leisure_fea_lis = ['leisure_' + str(i) for i in range(1, 13)]
data['leisure_sum'] = data[leisure_fea_lis].sum()  # skew

#
public_service_fea_lis = ['public_service_' + str(i) for i in range(1, 10)]
data['public_service_'] = data[public_service_fea_lis].sum()  # skew

data['city_public_service__mean'] = data.groupby(['city'])['public_service_'].transform('mean').values
data['public_service_cit'] = data['public_service_'] - data['city_public_service__mean']

#
trust_fea_lis = ['trust_' + str(i) for i in range(1, 10)]
data['trust_'] = data[trust_fea_lis].sum()  # skew

#   
data['survey_edu_age'] = 2015 - data['join_party']
data['survey_edu_age'].fillna(0, inplace=True)
del data['join_party']
del data['property_other']
del data['id']
del data['birth']
del data['edu_other']
# del data['gender']

data['urban'] = 2 - data['survey_type']
data['rural'] = data['survey_type'] - 1

# del data['survey_type']
data['hukou_1'] = data['hukou']
data['hukou_2'] = data['hukou'] - 1
data['hukou_3'] = data['hukou'] - 2
data['hukou_4'] = data['hukou'] - 3
data['hukou_5'] = data['hukou'] - 4
data['hukou_6'] = data['hukou'] - 5
data['hukou_7'] = data['hukou'] - 6
data.loc[data['hukou_1'] != 1, 'hukou_1'] = 0
data.loc[data['hukou_2'] != 1, 'hukou_2'] = 0
data.loc[data['hukou_3'] != 1, 'hukou_3'] = 0
data.loc[data['hukou_4'] != 1, 'hukou_4'] = 0
data.loc[data['hukou_5'] != 1, 'hukou_5'] = 0
data.loc[data['hukou_6'] != 1, 'hukou_6'] = 0
data.loc[data['hukou_7'] != 1, 'hukou_7'] = 0

# data['man'] = 2 - data['gender']
# data['woman'] = data['gender'] - 1
# data['hanzu'] =
# data['shaoshu'] =


# data['edu_1'] = data['edu_status']
# data['edu_2'] = data['edu_status'] - 1
# data['edu_3'] = data['edu_status'] - 2
# data['edu_4'] = data['edu_status'] - 3
#
# data.loc[data['edu_1'] != 1,'edu_1'] = 0
# data.loc[data['edu_2'] != 1,'edu_2'] = 0
# data.loc[data['edu_3'] != 1,'edu_3'] = 0
# data.loc[data['edu_4'] != 1,'edu_4'] = 0
##del data['edu_status']
#


# newdf1 = data.iloc[:,202:].values
# newdf2 = data.iloc[:,200:202].values
# newdf = np.zeros([newdf1.shape[0],14])
#
# for i in range(newdf1.shape[0]):
#    newdf[i] = (newdf1[i].reshape([-1,1]) * newdf2[i].reshape([-1,1]).T).reshape(-1)
#    
# newdf = pd.DataFrame(newdf)
# data = pd.concat([data,newdf], axis = 1)
#
# del data['hukou_1']
# del data['hukou_2']
# del data['hukou_3']
# del data['hukou_4']
# del data['hukou_5']
# del data['hukou_6']
# del data['hukou_7']
# del data['urban']
# del data['rural']

data.fillna(-1, inplace=True)
print('shape', data.shape)
data.head()

#
use_fea = [clo for clo in data.columns if clo != 'survey_time' and data[clo].dtype != object]

features = data[use_fea].columns
X_train = data[:train_shape][use_fea].values
y_train = target
X_test = data[train_shape:][use_fea].values

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
#

from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, f1_score
import lightgbm as lgb
import xgboost as xgb
import os
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, RepeatedKFold
import logging

##### lgb
param = {
    'num_leaves': 80,
    'min_data_in_leaf': 40,
    'objective': 'regression',
    'max_depth': -1,
    'learning_rate': 0.04,
    "min_child_samples": 30,
    "boosting": "gbdt",
    "feature_fraction": 0.9,
    "bagging_freq": 2,
    "bagging_fraction": 0.9,
    "bagging_seed": 2029,
    "metric": 'mse',
    "lambda_l1": 0.25,
    "lambda_l2": 0.2,
    "verbosity": -1}
folds = KFold(n_splits=10, shuffle=True, random_state=1016)  # StratifiedKFold?   KFold
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros(len(X_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
    val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                    early_stopping_rounds=200)
    oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))

##### xgb
xgb_params = {'eta': 0.05, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 4}

X_train_use = data[use_fea][:train_shape].values
X_test_use = data[use_fea][train_shape:].values

folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_xgb = np.zeros(train_shape)
predictions_xgb = np.zeros(len(X_test_use))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
    print("fold n{}".format(fold_ + 1))
    trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
    val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                    verbose_eval=100, params=xgb_params)
    oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
    predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, target)))

##############newmodel######################
from catboost import Pool, CatBoostRegressor
# cat_features=[0,2,3,10,11,13,15,16,17,18,19]
from sklearn.model_selection import train_test_split

X_train_use1 = data[use_fea][:train_shape].values
X_test_use = data[use_fea][train_shape:].values

kfolder = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_cb = np.zeros(train_shape)
predictions_cb = np.zeros(len(X_test_use))
kfold = kfolder.split(X_train, y_train)
fold_ = 0
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_train, y_train, test_size=0.3, random_state=2019)
for train_index, vali_index in kfold:
    print("fold nw{}".format(fold_))
    fold_ = fold_ + 1
    k_x_train = X_train[train_index]
    k_y_train = y_train[train_index]
    k_x_vali = X_train[vali_index]
    k_y_vali = y_train[vali_index]
    cb_params = {
        'n_estimators': 100000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'learning_rate': 0.03,
        'depth': 5,
        'use_best_model': True,
        'subsample': 0.6,
        'bootstrap_type': 'Bernoulli',
        'reg_lambda': 3
    }
    model_cb = CatBoostRegressor(**cb_params)
    # train the model
    model_cb.fit(k_x_train, k_y_train, eval_set=[(k_x_vali, k_y_vali)], verbose=100, early_stopping_rounds=50)
    oof_cb[vali_index] = model_cb.predict(k_x_vali, ntree_end=model_cb.best_iteration_)
    predictions_cb += model_cb.predict(X_test, ntree_end=model_cb.best_iteration_) / kfolder.n_splits

print("CV score: {:<8.8f}".format(mean_squared_error(oof_cb, y_train)))

# AdaBoost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

kfolder = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_svr = np.zeros(train_shape)
predictions_svr = np.zeros(len(X_test_use))
for fold_, (t, v) in enumerate(kfolder.split(X_train, y_train)):
    print("fold nw{}".format(fold_))
    k_x_train = X_train[t]
    k_y_train = y_train[t]
    k_x_vali = X_train[v]
    k_y_vali = y_train[v]
    # clf4 = AdaBoostRegressor(BayesianRidge(lambda_1=0.3, lambda_2=0.1), n_estimators=6, learning_rate=0.1,
    #                          loss='linear', random_state=2018)
    # clf4 = AdaBoostRegressor(Ridge(), n_estimators=5, learning_rate=0.1, loss='square', random_state=2017)
    # clf4 = AdaBoostRegressor(SVR(), n_estimators=5, learning_rate=0.1, random_state=2001)
    clf4 = SVR()
    clf4.fit(k_x_train, k_y_train)
    oof_svr[v] = clf4.predict(k_x_vali)
    predictions_svr += clf4.predict(X_test) / kfolder.n_splits
print("SVR MES:", mean_squared_error(y_train, oof_svr))

kfolder = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_rig = np.zeros(train_shape)
predictions_rig = np.zeros(len(X_test_use))
for fold_, (t, v) in enumerate(kfolder.split(X_train, y_train)):
    print("fold nw{}".format(fold_))
    k_x_train = X_train[t]
    k_y_train = y_train[t]
    k_x_vali = X_train[v]
    k_y_vali = y_train[v]
    clf4 = Ridge(normalize=True)
    clf4.fit(k_x_train, k_y_train)
    oof_rig[v] = clf4.predict(k_x_vali)
    predictions_rig += clf4.predict(X_test) / kfolder.n_splits
print("Ridge MES:", mean_squared_error(y_train, oof_rig))

#
train_stack = np.vstack([oof_lgb, oof_xgb, oof_cb, oof_svr, oof_rig]).transpose()
test_stack = np.vstack([predictions_lgb, predictions_xgb, predictions_cb, predictions_svr, predictions_rig]).transpose()

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()#(lambda_1=3, lambda_2=0.1)
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

print(mean_squared_error(target.values, oof_stack))
