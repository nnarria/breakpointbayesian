#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:41:07 2022

@author: nnarria
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#%%
##############################################################################
# Makes a plot the true series with fitted curve and 
# selected breakpoints.
##############################################################################

# INPUTS
# resMH_sumgamma: for the breakpoints, for component l: number of iterations 
# during which a difference between the segmentation part at time l and the 
# segmentation part at time (l-1) was non null (during post-burn-in).
# resMH_sumr: for functions from the dictionary, for component l: number of 
# iterations during which the function j from the dictionary was included 
# in the functional part (during post-burn-in).
# itertot: Total number of iterations to be used in both M-H and Gibbs Sampler
# burnin: Number of iterations that will not be considered
# data_serie: data series to which you want to detect change points
# breakpoints: index in which breakpoints are located
# mean_data_serie: average of the values of the unnormalized series
# sd_data_serie: sd of the values of the unnormalized series
# threshold_bp: probability threshold from which the change points 
# are selected
# threshold_fnc: probability threshold from which the functions are selected
# reconstructiontot: reconstructionmu + reconstructionf
# path: absolute path and file name to save image
# save_fig: save or not the plot to a file

def draw_plot(resMH_sumgamma, resMH_sumr, itertot, burnin, data_serie, 
              breakpoints, mean_data_serie, sd_data_serie, 
              threshold_bp, threshold_func, reconstructiontot, path, 
              title, xlabel, ylabel, save_fig=False):
        
    
    nPlot1 = len(resMH_sumgamma)    
    fig_a, ax = plt.subplots()    
    
    simu_01_ = data_serie * sd_data_serie + (mean_data_serie)
    reconstruction_ = reconstructiontot.reshape(-1,1)
    reconstruction_ = reconstruction_ * sd_data_serie + mean_data_serie

    ax.plot(np.arange(nPlot1), simu_01_, color='black', ls='solid', lw=2, 
               label='Simulated series')
    ax.plot(np.arange(nPlot1), reconstruction_, color='#ff9000', lw=2, 
               ls=(0, (5, 5)), label='Estimated expectation')   
    
    for i in breakpoints[1:]:
        ax.axvline(x=i, color='red', linewidth=1)
    
    # where some data has already been plotted to ax
    handles, labels = ax.axes.get_legend_handles_labels()
    #Break points
    blue_line = mlines.Line2D([],[], color='red', label='Breakpoints selected') 
    handles.append(blue_line) 


    ax.legend(handles=handles)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    plt.tight_layout()
    plt.show()
    
    if save_fig == True:
        fig_a.savefig(path, dpi=200)
    
          
          
          