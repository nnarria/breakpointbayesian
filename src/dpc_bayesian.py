#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 00:22:47 2020

@author: nnarria
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import segmentationFunctionalEffect as sfe
import matplotlib.lines as mlines
import os


#%%
# Definir directorio de trabajo
WORK_DIRECTORY = '/Users/nnarria/Documents/magister_estadistica_2019/memoria/work/github/nnarria/breakpointbayesian/'
os.chdir(WORK_DIRECTORY)

#%%
# load dictionary
data_fnct = pd.read_csv(WORK_DIRECTORY+'/data/Fmatrix.csv', decimal='.', 
                        delimiter=',')
data_fnct = data_fnct.iloc[:, 1:] # Excluir primera columnas (idx row)
Fmatrix = np.array(data_fnct)


# load serie example
data_serie = pd.read_csv(WORK_DIRECTORY+'/data/dataExample1.csv', decimal='.', 
                        delimiter=',')
data_serie = np.array(data_serie)

#%%
## parametros iniciales
itertot = 20000
burnin = 5000
lec1 = 50
lec2 = 50
nbseginit = 3
nbfuncinit = 3
nbtochangegamma = 2
nbtochanger = 2
n = len(data_serie)

# Probabilidad de ser punto de cambio para cada punto de la serie (0.01)
Pi = np.concatenate([np.array([1.00]), np.repeat(0.01, n-1)], axis=0)
    
# Probabilidad que una funcion sea seleccionada
eta = np.concatenate([[1], np.repeat(0.01, Fmatrix.shape[1]-1)])

# umbral de corte para punto de cambio y funciones
threshold_bp = 0.5
threshold_fnc = 0.5

result_ = sfe.dbp_with_function_effect(
        Fmatrix, data_serie, itertot, burnin, lec1, lec2,
        nbseginit, nbfuncinit, nbtochangegamma, nbtochanger, Pi, eta,
        threshold_bp, threshold_fnc, printiter=False)


#%%
##
# plot result
##


resMH = result_[0]
reconstructiontot = result_[7]

# break points simulated in serie
truebreakpoints = np.array([20,31,43,100])

fig_a = plt.figure(constrained_layout=True)
gs = fig_a.add_gridspec(2, 2)
f_ax1 = fig_a.add_subplot(gs[0,0])
nPlot1 = len(resMH["sumgamma"])
f_ax1.plot(np.arange(nPlot1), resMH["sumgamma"]/(itertot-burnin), 
           marker='o', markersize=3, color='black', linestyle='none')
f_ax1.plot(np.arange(nPlot1), np.repeat(0.5, nPlot1), color='red', 
           linewidth=1)
f_ax1.set_title('Breakpoints selection')
f_ax1.set_ylabel('Posterior probabilities')
    
f_ax2 = fig_a.add_subplot(gs[0,1])
nPlot2 = len(resMH["sumr"])
f_ax2.plot(np.arange(nPlot2), resMH["sumr"]/(itertot-burnin), 
           marker='o', markersize=3, color='black', linestyle='none')
f_ax2.plot(np.arange(nPlot2), np.repeat(0.5, nPlot2), color='red', 
           linewidth=1)
f_ax2.set_ylabel('Posterior probabilities')
f_ax2.set_title('Functions selection')
      
f_ax3 = fig_a.add_subplot(gs[1, :])
f_ax3.plot(data_serie, color='black', ls='solid', lw=2, label='True serie')
f_ax3.plot(reconstructiontot.reshape(-1,1), color='#ff9000', lw=2, 
           ls=(0, (5, 5)), label='Reconstruction')
    
xmin, xmax, ymin, ymax = f_ax3.axis()
for i in np.arange(len(truebreakpoints)-1):
    f_ax3.axvline(x=truebreakpoints[i], color='red', linewidth=1)   

# where some data has already been plotted to ax
handles, labels = f_ax3.get_legend_handles_labels()
red_line = mlines.Line2D([],[], color='red', label='Simulated break-points')

# handles is a list, so append manual patch
handles.append(red_line) 

f_ax3.legend(handles=handles) 
f_ax3.set_xlabel('Time')
