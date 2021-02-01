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
import os
import matplotlib.dates as mdates
import utilPlot as uplot


#%%
# Definir directorio de trabajo
WORK_DIRECTORY = '/Users/nnarria/Documents/magister_estadistica_2019/memoria/work/github/nnarria/breakpointbayesian/'
os.chdir(WORK_DIRECTORY)

#%%
# Para datos simulados
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
# para los datos USD/CLP
## carga de la serie de datos
data_fnct_dp = pd.read_csv(WORK_DIRECTORY+'/data/DolarPeso.txt', decimal='.', 
                        delimiter='\t')
data_fnct_dp['Fecha'] = pd.to_datetime(data_fnct_dp['Fecha'], 
                                       format='%d-%m-%Y')
# filtrar fechas de la serie
data_fnct_dp = data_fnct_dp[(data_fnct_dp['Fecha'] >= '20180101') & 
          (data_fnct_dp['Fecha'] <= '20200901')]


# 14-10-2019 estallido social el comienzo idx=441
# 18-10-2019 estallido social radicalizacion del movimiento idx=445
# 25-10-2019 estallido social la marcha mas grande la historia idx=450
# 03-03-2020 1er caso COVID idx=537
# 11-03-2020 1er caso COVID idx=543
# mostrar el valor dolar-peso para cada una de las fechas especiales
data_fnct_dp[data_fnct_dp['Fecha'].dt.strftime('%Y-%m-%d') == '2019-10-14']
data_fnct_dp[data_fnct_dp['Fecha'].dt.strftime('%Y-%m-%d') == '2019-10-18']
data_fnct_dp[data_fnct_dp['Fecha'].dt.strftime('%Y-%m-%d') == '2019-10-25']
data_fnct_dp[data_fnct_dp['Fecha'].dt.strftime('%Y-%m-%d') == '2020-03-03']
data_fnct_dp[data_fnct_dp['Fecha'].dt.strftime('%Y-%m-%d') == '2020-03-11']

# se ordena la serie y se resetea el index
data_fnct_dp = data_fnct_dp.sort_values(by='Fecha')
data_fnct_dp = data_fnct_dp.reset_index(drop=True)

data_serie = np.array(data_fnct_dp['Valor'])
data_serie = data_serie[~np.isnan(data_serie)]
data_serie.shape


#%%
# plot serie de datos USD-CLP
fig, ax = plt.subplots(figsize=(6, 6))

# Add x-axis and y-axis
ax.plot(data_fnct_dp['Fecha'], data_fnct_dp['Valor'], color='black', 
        linestyle='solid', linewidth=1, label='USD/CLP - Dólar Peso chileno')

# Set title and labels for axes
ax.set(xlabel="Día", ylabel="Peso chileno", 
       title="USD/CLP - Dólar Peso chileno")
ax.set_xlabel("Día")
ax.set_ylabel("Peso chileno")

# Define the date format
date_form = mdates.DateFormatter("%m-%Y")
ax.xaxis.set_major_formatter(date_form)

ax.axvline('2019-10-14', 
           label='14 de Octubre 2019 comienzo estallido social', 
           linestyle='dotted', color='black', linewidth=1)

ax.axvline('2019-10-18', 
           label='18 de Octubre 2019 generalización estallido social', 
           linestyle='dotted', color='black', linewidth=0.9)

ax.axvline('2020-03-03', label='03 de Marzo 2020 1er caso Covid-19 en Chile', 
           color='black', linestyle='dashed', linewidth=1)

ax.axvline('2020-03-11', label='11 de Marzo 2020 OMS declara pandemia', 
           color='black', linestyle='dashed', linewidth=0.9)
plt.legend()

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)


#%%
# se escalan los datos con media=0 y sd=1
data_serie_mean = np.mean(data_serie)
data_serie_sd = np.std(data_serie)
data_serie = (data_serie - data_serie_mean)/data_serie_sd
data_serie = data_serie.reshape(len(data_serie),1)


# load dictionary para datos USD/CLP
data_fnct = pd.read_csv(WORK_DIRECTORY+'/data/Fmatrix655x248.csv', decimal='.', 
                        delimiter=',') # 205 B-splines con 5o nodos

data_fnct = data_fnct.iloc[:, 1:]
Fmatrix = np.array(data_fnct)
Fmatrix.shape


#%%
## parametros iniciales
itertot = 160000
burnin = 40000
lec1 = 50
lec2 = 50
nbseginit = 3
nbfuncinit = 3
nbtochangegamma = 1
nbtochanger = 1
n = len(data_serie)

# prob. priori a ser asignado a los puntos especiales
tmpPi_ = 0.3 #0.001#0.3

# Probabilidad de ser punto de cambio para cada punto de la serie (0.01)
Pi = np.concatenate([np.array([1.00]), np.repeat(0.01, n-1)], axis=0)

# ESocial
Pi[445] = tmpPi_

# Covid-19
Pi[543] = tmpPi_

    
# Probabilidad que una funcion sea seleccionada
eta = np.concatenate([[1], np.repeat(0.01, Fmatrix.shape[1]-1)])

# umbral de corte para punto de cambio y funciones
threshold_bp = 0.6
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

# Puntos de cambio según umbral
breakpoints_bp = np.where(resMH["sumgamma"]/(itertot-burnin) > threshold_bp)[0]

# dibujar resultados
uplot.draw_plot(resMH["sumgamma"], resMH["sumr"], itertot, burnin, 
          data_serie, breakpoints_bp, data_fnct_dp['Fecha'], data_serie_mean, 
          data_serie_sd, threshold_bp, threshold_fnc, result_[7], 
          WORK_DIRECTORY, showDate=True, save_fig=False)

#%%




