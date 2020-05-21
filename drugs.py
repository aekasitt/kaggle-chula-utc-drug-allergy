#!/usr/bin/python
# coding:utf-8
# Copyright (C) 2019-2020 All rights reserved.
# FILENAME:  drug_allergy.py
# VERSION: 	 1.0
# CREATED: 	 2020-05-06 19:20
# AUTHOR: 	 Aekasitt Guruvanich <aekazitt@gmail.com>
# DESCRIPTION:
#
# HISTORY:
#*************************************************************
import re
from math import sqrt
### Third-Party Packages ###
from numpy import log
from pandas import read_csv, get_dummies
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
### Local Modules ###
from helpers.logger import Logger
from helpers.pickler import Pickler

def drug_allergy():
  ### Initiate Logger Instance ###
  logger = Logger.get_instance('drug_allergy')
  ### Initiate Pickler Instance ###
  pickler = Pickler.get_instance()
  data = pickler.load('drugs')
  if data is None:
    data = read_csv('data/drugs.csv')
    pickler.save(data, 'drugs')
  logger.info(f'Test Data: {data.shape}')
  logger.info(f'Test Data Columns: {data.columns}')

  ### Column Descriptions ###
  # ELISpot_Control is the ELISpot test result for the POSITIVE CONTROL (i.e., we expect to see strong response)
  # ELISpot_Result is the ELISpot test result for SUSPECTED DRUG (i.e., this is the result that indicate whether the patient would be allergic to that drug)
  # NARANJO_Category is ORDINAL.
  # Exposure_Time is the amount of times since the patient has taken the drug until the ELISpot test date
  # Suspicion_Score is the suspicion level of the drug (1 = suspected drug, 2 = similar to suspected drug, 3 = negative control). This is ORDINAL.
  # Allergic_Reaction_Group is the severity of patient's allergic reaction. This is ORDINAL.
  # Drug_Group is CATEGORICAL.
  # Drug_Rechallenge_Result is the ground truth of this dataset that we want to predict.

  ### Checks for null values and find the percentage of null values that we have ###
  logger.info(f'Columns with Null Data:\n{data.isnull().any()}')
  logger.info(f'Percentage of Null Data:\n{data.isnull().sum() / data.shape[0]}')

  ### Drop Duplicates ###
  data.drop_duplicates(inplace=True)

  ### Impute Underlying Conditions with Population Mode ###
  data.Underlying_Condition_A.fillna(data.Underlying_Condition_A.mode().iloc[0], inplace=True)
  data.Underlying_Condition_D.fillna(data.Underlying_Condition_D.mode().iloc[0], inplace=True)
  data.Underlying_Condition_E.fillna(data.Underlying_Condition_E.mode().iloc[0], inplace=True)

  ### Create Dummies for Categorical Indenpendent Variables ###
  for column in data.columns:
    matched = re.match(r'\w+_Group', column)
    if matched is not None:
        data = data.join(get_dummies(data[column], prefix=column, dummy_na=True))
        data.drop([column], axis=1, inplace=True)

  ### Naranjo Category and Naranjo Score ###
  dummy_naranjo = get_dummies(data['Naranjo_Category'], prefix='Naranjo_Category')
  naranjo = dummy_naranjo.mul(data.Naranjo_Score.fillna(0), axis=0)
  data = data.join(naranjo)
  data.drop(['Naranjo_Category', 'Naranjo_Score'], axis=1, inplace=True)

  ### Fills Exposure Time null rows with 0 ###
  data.Exposure_Time.fillna(0, inplace=True)

  ### ELISpot_Control is the ELISpot test result for the POSITIVE CONTROL (i.e., we expect to see strong response) ###
  data['ELISpot_Control_Log'] = log(data[['ELISpot_Control']])
  data.drop(['ELISpot_Control'], axis=1, inplace=True)

  # Suspicion_Score is the suspicion level of the drug
  # 1 = suspected drug
  # 2 = similar to suspected drug
  # 3 = negative control).
  # This is ORDINAL.
  suspicion = get_dummies(data['Suspicion_Score'], prefix='Suspicion_Score')
  suspicion.rename(columns={
      'Suspicion_Score_1': 'Suspicion_Level_Suspected',
      'Suspicion_Score_2': 'Suspicion_Level_Near_Suspected',
      'Suspicion_Score_3': 'Suspicion_Level_Negative_Control'
  }, inplace=True)

  ### To Merge or not to merge between Suspected and Near_Suspected
  suspicion['Suspicion_Level_Suspected'] = suspicion['Suspicion_Level_Suspected'] + suspicion['Suspicion_Level_Near_Suspected']
  suspicion.drop(['Suspicion_Level_Near_Suspected'], axis=1, inplace=True)
  data = data.join(suspicion)
  data.drop(['Suspicion_Score'], axis=1, inplace=True)

  logger.info(f'Test Data: {data.shape}')
  logger.info(f'Test Data Columns: {data.columns}')
  fold = 5
  xgb_params = dict( \
      booster= ['gbtree'],
      colsample_bytree= [0.9],
      learning_rate= [0.01, 0.05],
      n_estimators= [100, 300],
      max_delta_step= [0], # range: (0, infinity), defaults: 0
      max_depth= [6],      # range: (0, infinity), defaults: 6
      min_child_weight= [1, 5, 10],
      silent= [True],
      subsample= [0.7],       # range: (0, 1)
      reg_alpha= [0],         # L1 regularization term on weights
      reg_lambda= [1]         # L2 regularization term on weights
  )
  gsv = GridSearchCV(estimator=XGBClassifier(), param_grid=xgb_params, \
    scoring='neg_mean_absolute_error', verbose=10, n_jobs=15, cv=fold)
  trainable_data = data[data.Drug_Rechallenge_Result.notna()]
  x_train = trainable_data.drop(['Patient_ID', 'Drug_Rechallenge_Result'], axis=1)
  y_train = trainable_data['Drug_Rechallenge_Result']
  x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=.8, stratify=y_train)
  gsv.fit(x_train, y_train)
  model = gsv.best_estimator_
  logger.info(model)

  y_predict = model.predict(x_test)
  num_wrong_predictions = (y_predict != y_test).sum()
  r2 = r2_score(y_test, y_predict)
  mse = mean_squared_error(y_test, y_predict)
  rmse = sqrt(mse)
  logger.info(f'Number of Wrong Predictions: {num_wrong_predictions} / {len(y_predict)}')
  logger.info(f'R2: {r2:.4f}')
  logger.info(f'Mean Squared Error: {mse:.4f}')
  logger.info(f'Root Mean Squared Error: {rmse:.4f}')

  y_predict = model.predict(data.drop(['Patient_ID', 'Drug_Rechallenge_Result'], axis=1))
  results = data[['Patient_ID', 'Drug_Rechallenge_Result']]
  results.loc[:, 'Predicted_Drug_Rechallenge_Result'] = y_predict
  results.to_csv('results.csv', index=False)

  Logger.release_instance()
  Pickler.release_instance()

if __name__ == '__main__':
  drug_allergy()
