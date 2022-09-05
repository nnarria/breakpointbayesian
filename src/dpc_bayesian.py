#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 02:46:46 2022

@author: nnarria
"""

##############################################################################
# Sequence of steps to find the breakpoints from a given time series. 
# Starting with the preparation of the time series, the special dates are 
# defined, probabilities are assigned for prior knowledge, the methodology is 
# applied to find the breakpoints considering previously a series of 
# hyperparameters that must be fixed and finally the results are presented by 
# means of a plot with the probabilities of each point of the series of being 
# or not a change point, together with the probabilities to select the 
# functions that better adjust, also is shown in another plot the original 
# series with the adjusted series and the selected breakpoints.
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dpc_segmentationFunctionalEffect as sfe
import os
import matplotlib.dates as mdates
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
# marking relevant dates. USD/CLP
######################################
print('Section: Load time series to be submitted\n')

data_fnct_dp = pd.read_csv(WORK_DIRECTORY+'/data/DollarCLP.txt', decimal='.', 
                        delimiter='\t')
data_fnct_dp['Date'] = pd.to_datetime(data_fnct_dp['Date'], 
                                       format='%d-%m-%Y')
data_fnct_dp['Value'] = pd.to_numeric(data_fnct_dp['Value'])

# filter data
data_fnct_dp = data_fnct_dp[(data_fnct_dp['Date'] >= '20180101') & 
          (data_fnct_dp['Date'] <= '20220331')]

# sorted data by Fecha asc
data_fnct_dp = data_fnct_dp.sort_values(by='Date')
data_fnct_dp = data_fnct_dp.reset_index(drop=True)


# 1.- 14-10-2019 Estallido social el comienzo, idx=441
# 2.- 18-10-2019 Estallido social generalización del movimiento, idx=445
# 3.- 03-03-2020 1er caso COVID, idx=537
# 4.- 11-03-2020 OMS declara pandemia mundial, idx=543
# 5.- 20-12-2021 2do día luego de segunda vuelta elecciones presidenciales en 
# Chile, idx=986
# 6.- 24-02-2022 Invasión rusa de Ucrania 2022, idx=1033
# 7.- 11-03-20222 Traspaso de mando al nuevo presidente de la republica 
# idx=1044
special_date = np.array(['2019-10-14', '2019-10-18', '2020-03-03', 
                         '2020-03-11', '2021-12-20', '2022-02-24', 
                         '2022-03-11'])
special_date_idx = data_fnct_dp[
    data_fnct_dp['Date'].dt.strftime('%Y-%m-%d').isin(special_date)].index
special_date_idx = special_date_idx.values

# list idx special_date
special_date_idx

# Load USD/CLP series values only
data_serie = np.array(data_fnct_dp['Value'])

# check for nan values in the values
np.argwhere(np.isnan(data_serie))
data_serie.shape


# Se estandarizan los datos de la serie
data_serie_mean = np.mean(data_serie)
data_serie_sd = np.std(data_serie)
data_serie = (data_serie - data_serie_mean)/data_serie_sd


#%%
######################################
# Time series plot with relevant dates
######################################
print('Section: Time series plot with relevant dates\n')

path_fig = WORK_DIRECTORY + 'images/'

# Create figure and plot space
fig, ax = plt.subplots(figsize=(15, 9))

ax.set_aspect(2.9) 

# Add x-axis and y-axis
ax.plot(data_fnct_dp['Date'], data_fnct_dp['Value'], color='black', 
        linestyle='solid', linewidth=1, label='USD/CLP - Dollar Chilean peso')

# Set title and labels for axes
ax.set(xlabel="Day",
       ylabel="Chilean peso",
       title="USD/CLP - Dollar Chilean peso")
ax.set_xlabel("Day")
ax.set_ylabel("Chilean peso")

date_form = mdates.DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)

ax.axvline(dt.datetime(2019, 10, 14), 
           label='14 Oct 2019 beginning of a social outburst', 
           linestyle='dotted', color='black', linewidth=1)

ax.axvline(dt.datetime(2019, 10, 18), 
           label='18 Oct 2019 generalization of social outburst', 
           linestyle='dashed', color='black', linewidth=0.9)

ax.axvline(dt.datetime(2020, 3, 3), 
           label='03 Mar 2020 1st Covid-19 case in Chile', 
           color='black', linestyle='dashdot', linewidth=1)

ax.axvline(dt.datetime(2020, 3, 11), 
           label='11 Mar 2020 OMS declares pandemic', 
           color='orangered', linestyle='dotted', linewidth=0.9)

ax.axvline(dt.datetime(2021, 12, 20), 
           label='20 Dec 2021 Second round of presidential elections in Chile', 
           color='orangered', linestyle='dashed', linewidth=0.9)

ax.axvline(dt.datetime(2022, 2, 24), 
           label='24 Feb 2022 Russian invasion of Ukraine', 
           color='orangered', linestyle='dashdot', linewidth=0.9)

ax.axvline(dt.datetime(2022, 3, 11), 
           label='11 Mar 2022 Transfer of power, new president in Chile', 
           color='orangered', linestyle='dotted', linewidth=0.9)

plt.legend()

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

fig.savefig(path_fig + 'serie_dollar_clp_001.png', dpi=200)


#%%
########################################
# Load dictionary matrix
########################################
print('Section: Load dictionary matrix\n')

# 1     Constant function
# 64    Haar function
# 26    Fourier function 13 cos and 13 sin
# 1     Lineal function 
# 1     Cuadratic function
# 155   B-Spline 30 first order, 31 second, 32 third order and 62 fourth order
#-----
# 248   Total function
data_fnct = pd.read_csv(WORK_DIRECTORY +'/data/Fmatrix1059x248.csv', decimal='.', 
                        delimiter=',')

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
itertot         = 160000
burnin          = 40000
lec1            = 50
lec2            = 50
nbseginit       = 5
nbfuncinit      = 5
nbtochangegamma = 1
nbtochanger     = 1

# threshold breakpoints
threshold_bp    = 0.6 # breakpoint
threshold_fnc   = 0.5 # functions

# use or not data with apriori probabilities
flag_with_priori = True

# assign apriori probabilities
Pi = np.concatenate([np.array([1.00]), np.repeat(0.001, n-1)], axis=0)

priori_df = pd.concat([pd.DataFrame(special_date, columns=['Date']), 
           pd.DataFrame(special_date_idx, columns=['Idx'])], axis=1)
priori_df['ppriori'] = 0.001 # apriori probabilities default

priori_df.loc[0, 'ppriori'] = 0.3
priori_df.loc[1, 'ppriori'] = 0.3
priori_df.loc[2, 'ppriori'] = 0.3
priori_df.loc[3, 'ppriori'] = 0.3
priori_df.loc[4, 'ppriori'] = 0.3
priori_df.loc[5, 'ppriori'] = 0.3
priori_df.loc[6, 'ppriori'] = 0.3

if flag_with_priori:
    Pi[priori_df.Idx] = priori_df.ppriori


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
          data_serie, breakpoints_bp, data_fnct_dp['Date'], data_serie_mean, 
          data_serie_sd, threshold_bp, threshold_fnc, result_[7], 
          WORK_DIRECTORY+'/images/resMH_Dollar_CLP_test.png', show_date=True, 
          save_fig=True)


data_serie
data_serie_mean
np.mean(data_serie)

#%%
########################################
# Show dates of occurrence of breakpoints 
# and index of selected functions
########################################
print('Section: Show results\n')

# breakpoints and function selected
breakpoints_sel = np.where(resMH["sumgamma"]/(itertot-burnin) > 
                           threshold_bp)[0]
function_sel = np.where(resMH["sumr"]/(itertot-burnin) > 
                        threshold_fnc)[0]

# list dates breakpoints
data_fnct_dp.iloc[breakpoints_bp]

# list idx function
function_sel

