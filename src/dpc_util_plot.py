#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:18:42 2020

@author: nnarria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

#%%
##############################################################################
# Makes a plot showing the posterior probabilities of breakpoints and 
# functions, together with original series with fitted curve and 
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
# date_data_serie: dates of the data series
# mean_data_serie: average of the values of the unnormalized series
# sd_data_serie: sd of the values of the unnormalized series
# threshold_bp: probability threshold from which the change points 
# are selected
# threshold_fnc: probability threshold from which the functions are selected
# reconstructiontot: reconstructionmu + reconstructionf
# path: absolute path and file name to save image
# show_date: display or not the dates in the plot
# save_fig: save or not the plot to a file

def draw_plot(resMH_sumgamma, resMH_sumr, itertot, burnin, data_serie, 
              breakpoints, date_data_serie, mean_data_serie, sd_data_serie, 
              threshold_bp, threshold_func, reconstructiontot, path, 
              show_date=False, save_fig=False):
        
    idx_ = pd.to_datetime(date_data_serie, format='___')
    
    fig_a = plt.figure(constrained_layout=True)
    gs = fig_a.add_gridspec(2, 2)

    # for ax1 is later prob. for breakpoints
    nPlot1 = len(resMH_sumgamma)
    
    # for ax2 is prob. functions
    nPlot2 = len(resMH_sumr)
    
    if show_date == True:
        
        ### ax1
        f_ax1 = fig_a.add_subplot(gs[0,0])
        f_ax1.xaxis.set_minor_locator(ticker.MultipleLocator(115))
        f_ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        f_ax1.xaxis.set_major_locator(mdates.YearLocator())
        f_ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
        f_ax1.tick_params(axis='x', which='both', labelsize=10)
        f_ax1.tick_params(axis='y', labelsize=10)

        f_ax1.plot(idx_, resMH_sumgamma/(itertot-burnin), 
                   marker='o', markeredgecolor='black', markersize=3, 
                   color='white', linestyle='none')


        f_ax1.axhline(y=threshold_bp, color='red', linewidth=1)
        f_ax1.set_title('Breakpoints selection', size=14) 
        f_ax1.set_ylabel('Posteriori probabilities', fontsize='medium')


        ### ax2
        f_ax2 = fig_a.add_subplot(gs[0,1])
        f_ax2.tick_params(axis='x', which='both', labelsize=10)
        f_ax2.tick_params(axis='y', labelsize=10)

        f_ax2.plot(np.arange(nPlot2), resMH_sumr/(itertot-burnin), 
                   marker='o', markersize=3, markeredgecolor='black', 
                   color='white', linestyle='none')

        f_ax2.axhline(y=threshold_func, color='red', linewidth=1)
        f_ax2.set_ylabel('Posteriori probabilities', fontsize='medium')
        f_ax2.set_title('Functions selection', size=14)


        ### ax3
        f_ax3 = fig_a.add_subplot(gs[1, :])
        f_ax3.xaxis.set_minor_locator(ticker.MultipleLocator(115))
        f_ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        f_ax3.xaxis.set_major_locator(mdates.YearLocator())
        f_ax3.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))

        simu_01_ = data_serie * sd_data_serie + (mean_data_serie)
        reconstruction_ = reconstructiontot.reshape(-1,1)
        reconstruction_ = reconstruction_ * sd_data_serie + mean_data_serie

        f_ax3.plot(idx_, simu_01_, color='black', ls='solid', lw=2, 
                   label='USD/CLP series') # True serie
        f_ax3.plot(idx_, reconstruction_, color='#ff9000', lw=2, 
                   ls=(0, (5, 5)), label='Estimated serie') # Reconstruction    
    
        for i in breakpoints[1:]:
            f_ax3.axvline(x=idx_[i], color='red', linewidth=1)
    
    
        # where some data has already been plotted to ax
        handles, labels = f_ax3.get_legend_handles_labels()
        # Break points
        blue_line = mlines.Line2D([],[], color='red', 
                                  label='Change point')

        # handles is a list, so append manual patch
        handles.append(blue_line) 

        f_ax3.legend(handles=handles)
        f_ax3.set_title('USD/CLP series and estimated model', size=14)
        f_ax3.set_ylabel('Chilean peso', fontsize='large')
        f_ax3.set_xlabel('Day', fontsize='large')
    
    else:
        ### ax1
        f_ax1 = fig_a.add_subplot(gs[0,0])
        f_ax1.plot(np.arange(nPlot1), resMH_sumgamma/(itertot-burnin), 
                   marker='o', markeredgecolor='black', markersize=3, 
                   color='white', linestyle='none')

        f_ax1.axhline(y=threshold_bp, color='red', linewidth=1)
        f_ax1.set_title('Breakpoints selection')
        f_ax1.set_ylabel('Posteriori probabilities')


        ### ax2
        f_ax2 = fig_a.add_subplot(gs[0,1])
        f_ax2.plot(np.arange(nPlot2), resMH_sumr/(itertot-burnin), 
                   marker='o', markersize=3, markeredgecolor='black', 
                   color='white', linestyle='none')

        f_ax2.axhline(y=threshold_func, color='red', linewidth=1)
        f_ax2.set_ylabel('Posteriori probabilities')
        f_ax2.set_title('Functions selection')


        ### ax3
        f_ax3 = fig_a.add_subplot(gs[1, :])
        simu_01_ = data_serie * sd_data_serie + (mean_data_serie)
        reconstruction_ = reconstructiontot.reshape(-1,1)
        reconstruction_ = reconstruction_ * sd_data_serie + mean_data_serie

        f_ax3.plot(np.arange(nPlot1), simu_01_, color='black', ls='solid', 
                   lw=2, label='USD/CLP series')
        f_ax3.plot(np.arange(nPlot1), reconstruction_, color='#ff9000', lw=2, 
                   ls=(0, (5, 5)), label='Estimated series')   
    
        for i in breakpoints[1:]:
            f_ax3.axvline(x=i, color='red', linewidth=1)
    
        # where some data has already been plotted to ax
        handles, labels = f_ax3.get_legend_handles_labels()
        #Break points
        blue_line = mlines.Line2D([],[], color='red', label='Change points') 

        # handles is a list, so append manual patch
        handles.append(blue_line) 

        f_ax3.legend(handles=handles)
        f_ax3.set_title('USD/CLP series and estimated model')
        f_ax3.set_ylabel('Chilean peso')
        f_ax3.set_xlabel('Day')


    if save_fig == True:
        fig_a.savefig(path, dpi=200)
          
          
          
          
          
          
          