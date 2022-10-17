"""This is a script for visualisation and analysis of results from the replica A. Ure model"""

import pandas as pd
import matplotlib.pyplot as plt
from math import floor
from math import ceil
from model_func import coefs_and_metrics
import numpy as np
from sklearn.metrics import mean_absolute_error as mae

# Load in the model results data.
# mod_results = pd.read_excel("A.Ure replica results.xlsx")
mod_results = pd.read_excel("A.Ure replica results random #2.xlsx")
mod_results = mod_results[mod_results.columns[1:]]
# metrics = pd.read_excel("A.Ure replica coefficients and metrics.xlsx")
metrics = pd.read_excel("A.Ure replica coefficients and metrics random #2.xlsx")
# metrics = metrics[["R^2", "RMSE"]]
metrics = metrics[["R^2", "MAPE"]]

# Print model results to console to check if everything loaded in correctly.
print(mod_results)
print(metrics, '\n')

# Create variables that are easier to reference.
train_y = mod_results['Training Viscosity'].values
train_pred = mod_results['Training Prediction'].values
test_y = mod_results['Test Viscosity'].values
test_pred = mod_results['Test Prediction'].values


# Create a list of predicted means and standard deviations.
sorting_ind = np.flip(test_y.argsort())
sorted_y = test_y[sorting_ind[::-1]]
sorted_pred = test_pred[sorting_ind[::-1]]
step_size = len(sorted_y)/1000
real_means = []
pred_means = []
error_means = []
stds = []
error_visc = []

for ind in range(1000):
    real = sorted_y[int(ind*step_size): int((ind+1)*step_size)]
    pred = sorted_pred[int(ind*step_size): int((ind+1)*step_size)]
    error_visc.append(np.mean(sorted_y[int(ind*step_size): int((ind+1)*step_size)]))
    real_means.append(np.mean(real))
    pred_means.append(np.mean(pred))
    error_means.append(np.mean(real - pred))
    stds.append(np.std(real - pred))
real_means = np.array(real_means)
pred_means = np.array(pred_means)
error_means = np.array(error_means)
stds = np.array(stds)
error_visc = np.array(error_visc)

print(real_means)

# Create figures of Prediction vs. Experimental and overlay distribution violin plots.
fig1, ax = plt.subplots(nrows=1, ncols=2)
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
axes1 = [ax[0], ax[0].twinx()]
axes1[1].violinplot(train_y, points=80, showmeans=True, showextrema=True, bw_method=0.5, vert=False)
axes1[0].plot(train_y, train_pred, 'bo', linewidth=0.25)
axes1[0].plot(real_means, pred_means, '-', linewidth=1, color='yellowgreen')
axes1[0].plot([floor(min(train_y)), ceil(max(train_pred))], [floor(min(train_y)), ceil(max(train_pred))], 'r--',
              linewidth=1)
axes1[0].yaxis.set_ticks([tick for tick in range(floor(min(train_y)), ceil(max(train_pred))+1)])
axes1[0].xaxis.set_ticks([tick for tick in range(floor(min(train_y)), ceil(max(train_pred))+1)])
axes1[1].yaxis.set_ticks([])
ax[0].text(0.125, 0.95, '$R^2$ = {}'.format(round(metrics['R^2'][0], 2)), horizontalalignment='center',
           verticalalignment='center', transform=ax[0].transAxes, fontsize=14)
ax[0].set_title("Prediction of Training Viscosity", fontsize=18)
ax[0].set_xlim([4, 17])
ax[0].set_ylim([4, 17])
ax[0].set_xlabel("Experimental Viscosity (mPa s)", fontsize=16)
ax[0].set_ylabel("Predicted Viscosity (mPa s)", fontsize=16)
axes1[0].legend(['Training Set', 'Mean Prediction', 'True Line'], loc=3, fontsize=12)
axes1[1].legend(['Training Data Distribution'], loc=4, fontsize=12)


axes2 = [ax[1], ax[1].twinx()]
axes2[1].violinplot(test_y, points=80, showmeans=True, showextrema=True, bw_method=0.5, vert=False)
axes2[0].plot(test_y, test_pred, 'bo', linewidth=0.25)
axes2[0].plot(real_means, pred_means, '-', linewidth=1, color='yellowgreen')
axes2[0].plot([floor(min(train_y)), ceil(max(train_pred))], [floor(min(train_y)), ceil(max(train_pred))], 'r--',
              linewidth=1)
axes2[0].yaxis.set_ticks([tick for tick in range(floor(min(train_y)), ceil(max(train_pred))+1)])
axes2[0].xaxis.set_ticks([tick for tick in range(floor(min(train_y)), ceil(max(train_pred))+1)])
axes2[1].yaxis.set_ticks([])
# ax[1].text(1.375, 0.95, "RMSE = {} mPa s".format(round(metrics['RMSE'][0], 2)), horizontalalignment='center',
#            verticalalignment='center', transform=ax[0].transAxes, fontsize=14)
ax[1].text(1.375, 0.95, "MAPE = {} %".format(round(metrics['MAPE'][0]*100, 2)), horizontalalignment='center',
           verticalalignment='center', transform=ax[0].transAxes, fontsize=14)
ax[1].set_title("Prediction of Test Viscosity", fontsize=18)
ax[1].set_xlim([4, 17])
ax[1].set_ylim([4, 17])
ax[1].set_xlabel("Experimental Viscosity (mPa s)", fontsize=16)
ax[1].set_ylabel("Predicted Viscosity (mPa s)", fontsize=16)
axes2[0].legend(['Test Set', 'Mean Prediction', 'True Line'], loc=3, fontsize=12)
axes2[1].legend(['Test Data Distribution'], loc=4, fontsize=12)
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(error_visc, real_means, 'b-', linewidth=2, label="Experimental Viscosity")
ax2.plot(error_visc, pred_means, 'r-', linewidth=2, label="Predicted Viscosity")
ax2.plot(error_visc, (pred_means - stds), 'r--', linewidth=0.5, label="Prediction standard Deviation")
ax2.plot(error_visc, (pred_means + stds), 'r--', linewidth=0.5)
ax2.plot(error_visc, (real_means - 0.106), 'b--', linewidth=0.5, label="ASTM Uncertainty")
ax2.plot(error_visc, (real_means + 0.106), 'b--', linewidth=0.5)
ax2.set_title("Mean Experimental and predicted Viscosity", fontsize=16)
ax2.set_xlabel("Experimental Viscosity (mPa s)", fontsize=14)
ax2.set_ylabel("Predicted Viscosity (mPa s)", fontsize=14)
ax2.yaxis.set_ticks([tick for tick in range(4, 16)])
ax2.xaxis.set_ticks([tick for tick in range(4, 16)])
ax2.set_xlim([5, 15])
ax2.set_ylim([5, 15])
ax2.axvspan(8.5, 12.5, alpha=0.3, color='green')
ax2.axvspan(5, 8.5, alpha=0.3, color='red')
ax2.axvspan(12.5, 15, alpha=0.3, color='red')
ax2.legend()
plt.show()

coefs_and_metrics("A.Ure replica coefficients and metrics random #2.xlsx")
