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

def draw_plot(resMH_sumgamma, resMH_sumr, itertot, burnin, simu_01, 
              breakpoints, simu_01_date, simu_01_mean, simu_01_sd, 
              threshold_bp, threshold_func, reconstructiontot, path, 
              showDate=False, save_fig=False):
    

    
    #resMH_sumgamma=simu_01_resMH["sumgamma"]
    #resMH_sumr=simu_01_resMH["sumr"]
    #itertot=itertot
    #burnin=burnin
    #simu_01=simu_01
    #breakpoints=breakpoints
    #simu_01_date=data_fnct_dp['Fecha']
    #threshold_bp=0.98
    #threshold_func=0.5
    #reconstructiontot=reconstructiontot
    #path=path
    #showDate=False
    #show_fig=False
    
    
    idx_ = pd.to_datetime(simu_01_date, format='___')
    
    fig_a = plt.figure(constrained_layout=True)
    gs = fig_a.add_gridspec(2, 2)

    # para ax1 es prob. posterior para breakpoints
    nPlot1 = len(resMH_sumgamma)
    
    # para ax2 es prob. functions
    nPlot2 = len(resMH_sumr)
    
    if showDate == True:
        
        # ax.xaxis.set_major_locator(ticker.AutoLocator())
        # ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        
        ### ax1
        f_ax1 = fig_a.add_subplot(gs[0,0])
        # f_ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        f_ax1.xaxis.set_minor_locator(ticker.MultipleLocator(115))
        f_ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        f_ax1.xaxis.set_major_locator(mdates.YearLocator())
        #f_ax1.xaxis.set_major_locator(ticker.AutoLocator())
        f_ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
        f_ax1.tick_params(axis='x', which='both', labelsize=10)
        f_ax1.tick_params(axis='y', labelsize=10)

        #f_ax1.xaxis.set_tick_params(labelsize='x-small')
        f_ax1.plot(idx_, resMH_sumgamma/(itertot-burnin), 
                   marker='o', markeredgecolor='black', markersize=3, 
                   color='white', linestyle='none')


        f_ax1.axhline(y=threshold_bp, color='red', linewidth=1)
        f_ax1.set_title('Selección puntos de cambio', size=14) # Breakpoints selection
        f_ax1.set_ylabel('Probabilidad posteriori', fontsize='large') # Posterior probabilities


        ### ax2
        f_ax2 = fig_a.add_subplot(gs[0,1])
        f_ax2.tick_params(axis='x', which='both', labelsize=10)
        f_ax2.tick_params(axis='y', labelsize=10)

        f_ax2.plot(np.arange(nPlot2), resMH_sumr/(itertot-burnin), 
                   marker='o', markersize=3, markeredgecolor='black', 
                   color='white', linestyle='none')

        f_ax2.axhline(y=threshold_func, color='red', linewidth=1)
        f_ax2.set_ylabel('Probabilidad posteriori', fontsize='large') # Posterior probabilities
        f_ax2.set_title('Selección de funciones', size=14) # Functions selection


        ### ax3
        f_ax3 = fig_a.add_subplot(gs[1, :])
        # f_ax3.xaxis.set_minor_locator(mdates.MonthLocator())
        f_ax3.xaxis.set_minor_locator(ticker.MultipleLocator(115))
        f_ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
        f_ax3.xaxis.set_major_locator(mdates.YearLocator())
        f_ax3.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))

        simu_01_ = simu_01 * simu_01_sd + (simu_01_mean)
        reconstruction_ = reconstructiontot.reshape(-1,1)
        reconstruction_ = reconstruction_ * simu_01_sd + simu_01_mean

        f_ax3.plot(idx_, simu_01_, color='black', ls='solid', lw=2, 
                   label='Serie USD/CLP') # True serie
        f_ax3.plot(idx_, reconstruction_, color='#ff9000', lw=2, 
                   ls=(0, (5, 5)), label='Serie estimada') # Reconstruction    
    
        for i in breakpoints[1:]:
            f_ax3.axvline(x=idx_[i], color='red', linewidth=1)
    
    
        # where some data has already been plotted to ax
        handles, labels = f_ax3.get_legend_handles_labels()
        # Break points
        blue_line = mlines.Line2D([],[], color='red', 
                                  label='Puntos de cambio')

        # handles is a list, so append manual patch
        handles.append(blue_line) 

        f_ax3.legend(handles=handles)
        f_ax3.set_title('Serie USD/CLP y modelo estimado', size=14) # True serie and reconstruction
        f_ax3.set_ylabel('Peso chileno', fontsize='large') # Chilean Pesos
        f_ax3.set_xlabel('Día', fontsize='large') # day
    
    else:
        ### ax1
        f_ax1 = fig_a.add_subplot(gs[0,0])
        #f_ax1.xaxis.set_tick_params(labelsize='x-small')
        f_ax1.plot(np.arange(nPlot1), resMH_sumgamma/(itertot-burnin), 
                   marker='o', markeredgecolor='black', markersize=3, 
                   color='white', linestyle='none')

        f_ax1.axhline(y=threshold_bp, color='red', linewidth=1)
        f_ax1.set_title('Selección puntos de cambio') # Breakpoints selection
        f_ax1.set_ylabel('Probabilidad posteriori') # Posterior probabilities


        ### ax2
        f_ax2 = fig_a.add_subplot(gs[0,1])
        f_ax2.plot(np.arange(nPlot2), resMH_sumr/(itertot-burnin), 
                   marker='o', markersize=3, markeredgecolor='black', 
                   color='white', linestyle='none')

        f_ax2.axhline(y=threshold_func, color='red', linewidth=1)
        f_ax2.set_ylabel('Probabilidad posteriori') # Posterior probabilities
        f_ax2.set_title('Selección de funciones') # Functions selection


        ### ax3
        f_ax3 = fig_a.add_subplot(gs[1, :])
        simu_01_ = simu_01 * simu_01_sd + (simu_01_mean)
        reconstruction_ = reconstructiontot.reshape(-1,1)
        reconstruction_ = reconstruction_ * simu_01_sd + simu_01_mean

        f_ax3.plot(np.arange(nPlot1), simu_01_, color='black', ls='solid', 
                   lw=2, label='Serie USD/CLP') # True serie
        f_ax3.plot(np.arange(nPlot1), reconstruction_, color='#ff9000', lw=2, 
                   ls=(0, (5, 5)), label='Serie estimada') # Reconstruction    
    
        for i in breakpoints[1:]:
            f_ax3.axvline(x=i, color='red', linewidth=1)
    
        # where some data has already been plotted to ax
        handles, labels = f_ax3.get_legend_handles_labels()
        #Break points
        blue_line = mlines.Line2D([],[], color='red', label='Puntos de cambio') 

        # handles is a list, so append manual patch
        handles.append(blue_line) 

        f_ax3.legend(handles=handles)
        f_ax3.set_title('Serie USD/CLP y modelo estimado') # True serie and reconstruction
        f_ax3.set_ylabel('Peso chileno') # Chilean Pesos
        f_ax3.set_xlabel('Día') # day


    if save_fig == True:
        fig_a.savefig(path+'/resMH_pesoDolar.png', dpi=200)
          
          
          
          
          
          
          