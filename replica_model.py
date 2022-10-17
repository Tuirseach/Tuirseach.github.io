"""This is a script for replicating the training procedure described by A.Ure."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error as mape


# Load the model data
# dataset = pd.read_excel("A_Ure_replica_dataset.xlsx")
# dataset = pd.read_excel("A_Ure_replica_dataset_random_v4.xlsx")
dataset = pd.read_excel("A.Ure replica dataset random #2.xlsx")

# Select only the columns which will be used for modelling.
dataset = dataset[['MW', 'pCH3', 'pCH2', 'pCH', 'cyCH2', 'cyCH', 'alCH3', 'alCH', 'ArCH', 'ArC', 'rjC',
                  'Viscosity @ 243.15']]


# Define the descriptor headings and the target heading.
desc = ['MW', 'pCH3', 'pCH2', 'pCH', 'cyCH2', 'cyCH', 'alCH3', 'alCH', 'ArCH', 'ArC', 'rjC']
target = ['Viscosity @ 243.15']


# Split the data into training and test sets
train_X, test_X, train_y, test_y = train_test_split(dataset[desc], dataset[target], test_size=0.5, random_state=1)


# Define some model parameters before training.
regressor = linear_model.LinearRegression(fit_intercept=True, copy_X=True, positive=False)

# First replicating the approach described by Dr Ure.
backward_eliminator = sfs(regressor, n_features_to_select=5, direction='backward',
                          scoring='neg_root_mean_squared_error', cv=5)


# Select features by backward elimination and transform the descriptor data.
backward_eliminator.fit(train_X, train_y)
train_X_trans = backward_eliminator.transform(train_X)
test_X_trans = backward_eliminator.transform(test_X)
mask = backward_eliminator._get_support_mask()
selected = [feature for m, feature in zip(mask, desc) if m]


# Train a regression model on the selected features and calculate the score (R^2).
trained = regressor.fit(train_X_trans, train_y)
score = trained.score(train_X_trans, train_y)


# Predict the test set and calculate the root mean squared error (RMSE).
pred_y = trained.predict(test_X_trans)
train_pred = trained.predict(train_X_trans)
rmse = mean_squared_error(test_y, pred_y, squared=True)**(1/2)
pred_mape = mape(test_y, pred_y)

# Collect the data and output to an Excel file for analysis.
output = pd.DataFrame()
output['Training Viscosity'] = train_y
output['Training Prediction'] = train_pred
output['Test Viscosity'] = test_y.values
output['Test Prediction'] = pred_y

model = pd.DataFrame()
model['R^2'] = [score]
# model['RMSE'] = [rmse]
model['MAPE'] = [pred_mape]
model['Constant'] = trained.intercept_
for ind in range(len(selected)):
    model[selected[ind]] = [trained.coef_[0][ind]]

# output.to_excel('A.Ure replica results.xlsx')
# model.to_excel('A.Ure replica coefficients and metrics.xlsx')
output.to_excel('A.Ure replica results random #2.xlsx')
model.to_excel('A.Ure replica coefficients and metrics random #2.xlsx')
# output.to_excel('A_Ure_replica_results_random_v3.xlsx')
# model.to_excel('A_Ure_replica_coefficients_and_metrics_random_v3.xlsx')

print(model)

# # Train a model forced to use the same descriptors selected in Dr. A. Ure's model.
# train_X_force = train_X[['MW', 'pCH3', 'pCH2', 'cyCH2', 'ArC']]
# test_X_force = test_X[['MW', 'pCH3', 'pCH2', 'cyCH2', 'ArC']]
#
# trained_force = regressor.fit(train_X_force, train_y)
# score_force = trained_force.score(train_X_force, train_y)
#
# # Predict the test set and calculate the root mean squared error (RMSE).
# pred_y_force = trained_force.predict(test_X_force)
# train_pred_force = trained_force.predict(train_X_force)
# rmse_force = mean_squared_error(test_y, pred_y_force, squared=True)**(1/2)
#
# print(trained_force.intercept_)
# print(trained_force.coef_)
#
# import matplotlib.pyplot as plt
# test_pred = pd.DataFrame(pred_y_force, columns=['Viscosity'])
# train_pred = pd.DataFrame(train_pred_force, columns=['Viscosity'])
# metrics = {'RMSE': rmse_force, 'R^2': score_force}
#
# print(train_y, train_pred)
# print(test_y, test_pred)
