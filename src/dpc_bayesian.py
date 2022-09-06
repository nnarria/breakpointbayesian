#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 20:56:57 2022

@author: nnarria
"""

##############################################################################
# Sequence of steps to find the breakpoints from a given time series. 
# Starting with the preparation of the time series, the methodology is 
# applied to find the breakpoints considering previously a series of 
# hyperparameters that must be fixed and finally the results are presented
# shown the true series with the adjusted series and the selected 
# breakpoints.
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dpc_segmentation_functional_effect as sfe
import os
import dpc_util_plot as uplot
import datetime as dt


#%%
######################################
# Define working directory
######################################
print('Section: Define working directory\n')

WORK_DIRECTORY = '/Users/nnarria/Development/python_dev/breakpointbayesian/'
os.chdir(WORK_DIRECTORY)


#%%
######################################
# Load time series to be submitted 
# simulated data
######################################
print('Section: Load time series to be submitted\n')

data_fnct_dp = pd.read_csv(WORK_DIRECTORY+'/data/data_example1.txt', 
                           decimal='.', delimiter='\t', header=None)
data_fnct_dp.columns = ['Value']

# Load series values only
data_serie = np.array(data_fnct_dp['Value'])
data_serie.shape

# Se estandarizan los datos de la serie
data_serie_mean = np.mean(data_serie)
data_serie_sd = np.std(data_serie)
data_serie = (data_serie - data_serie_mean)/data_serie_sd


#%%
######################################
# Time series plot with simulated 
# breakpoints
######################################
print('Section: Time series plot\n')

path_fig = WORK_DIRECTORY + 'images/'
fig, ax = plt.subplots()

ax.plot(data_fnct_dp['Value'], color='black', 
        linestyle='solid', linewidth=1, label='Simulated serie')

ax.set(xlabel="Time",
       ylabel="Value",
       title="Simulated serie")

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
fig.savefig(path_fig + 'simulated_serie_0.png', dpi=200)


#%%
########################################
# Load dictionary matrix
########################################
print('Section: Load dictionary matrix\n')

# 1     Constant function
# 100    Haar function
# 20    Fourier function 10 cos and 10 sin
# 1     Lineal function 
# 1     Cuadratic function
#-----
# 123   Total function
data_fnct = pd.read_csv(WORK_DIRECTORY +'/data/Fmatrix_100x123.csv', 
                        decimal='.', delimiter=',')

data_fnct = data_fnct.iloc[:, 1:]
Fmatrix = np.array(data_fnct)
Fmatrix.shape

#%%
########################################
# Metropolis-Hastings
########################################
print('Section: Metropolis-Hastings\n')
print('Start: ' + str(dt.datetime.now()))

## init proceso 
M = 1 # one serie
n = len(data_serie)
    
##########################################################################
# algorithm Metropolis_Hastings to estimates the breakpoints of the 
# segmentation part and the functions present in the functional part
##########################################################################

# assign values to hyperparameters
itertot         = 20000
burnin          = 5000
lec1            = 50
lec2            = 50
nbseginit       = 5
nbfuncinit      = 5
nbtochangegamma = 2
nbtochanger     = 2

# threshold breakpoints
threshold_bp    = 0.6 # breakpoint
threshold_fnc   = 0.5 # functions

# assign apriori probabilities
Pi = np.concatenate([np.array([1.00]), np.repeat(0.001, n-1)], axis=0)

# probability that the dictionary function "j" is included in the 
# functional part. By convention eta[0] = 1
eta = np.concatenate([[1], np.repeat(0.01, Fmatrix.shape[1]-1)])


result_ = sfe.dbp_with_function_effect(
        Fmatrix, data_serie, itertot, burnin, lec1, lec2,
        nbseginit, nbfuncinit, nbtochangegamma, nbtochanger, Pi, eta,
        threshold_bp, threshold_fnc, printiter=False)


print('resMH calculated')
print('Finish: ' + str(dt.datetime.now()))


#%%
########################################
# Result plot
########################################
print('Section: Result plot\n')

resMH = result_[0]

# Change points according to threshold
breakpoints_bp = np.where(resMH["sumgamma"]/(itertot-burnin) > threshold_bp)[0]


# draw results
uplot.draw_plot(resMH["sumgamma"], resMH["sumr"], itertot, burnin, 
          data_serie, breakpoints_bp, data_serie_mean, data_serie_sd, 
          threshold_bp, threshold_fnc, result_[7], 
          WORK_DIRECTORY+'/images/resMH_0.png', 
          title='True and estimated functional part', xlabel='time', 
          ylabel='value', save_fig=True)


