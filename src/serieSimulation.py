# -*- coding: utf-8 -*-
"""
Create Fri Aug 07 17:23:45

@author: nnarria
"""

import numpy as np
import pandas as pd

##########################
### Simulation of series
##########################
# function to simulate M series, each one having their own breakpoints, but 
# all sharing the same number of segments, and the same functional part which 
# is specified in the article.

# INPUTS
# M: the number of series (if M>1, we are in case of multplie series)
# n: the number of points of the series
# K: the number of segments for each serie
# mu.choix: vector which gives the possible values for each segment of the 
# segmentation part of the series. The values of each segment are unifomly 
# sampled from this vector, and the value of a segment should not be equal to 
# the value of the preceding segment.
# varianceError: vector of variances for the error term
# pHaar: the positions of the Haar functions composing the functional part
# p1: A breakpoint should be positioned at a distance from the Haar functions 
# of at least p1
# p2: Each segment is at least of length p2

# OUTPUTS
# K: the number of segments for each serie
# muMat: matrix (M x n) which gives the value of the segmentation part of 
# each serie at each position
# tauMat: matrix (M x k.max) which gives the index of the breakpoints for each 
# series (K points for each series)
# errorMat: matrice (M x n) which gives the error term at every time point, 
# for each series
# biais: evaluation of the functional part at every time point
# Profile: data.frame which summarizes all the preceding informations in 
# columns: number (id) of the series, position in the grid of time points, 
# value of the segmentation part at this position, presence or not of a 
# breakpoint at this position, value of the functional part at this position, 
# error term at this position (one column for each specified variance of the 
# error term)

def simuSeries(M, n, K, muChoix, varianceError, pHaar, p1, p2):
    
    #### only 1 series with 4 segments, with a functional part, 100 points. 
    #We specify here only one variance for the error term only.
    #M = 1
    #n = 100
    #K = 4
    #muChoix = np.array([0, 1, 2, 3, 4, 5])
    #standard_deviation = np.array([0.1])
    #varianceError = np.power(standard_deviation, 2)
    #pHaar = np.array([10, 50, 60])
    #p1 = 3
    #p2 = 5
    #nbsimu = 1
    

    # Construction of the functional part f
    t = np.arange(n)
    A = 0.3
    Part1 = A*np.sin(2*np.pi*t/20)
    Part2 = np.zeros(n)
    t1 = pHaar[0]
    t2 = pHaar[1]
    t3 = pHaar[2]
    Part2[t1] = 1.5
    Part2[t2] = -2
    Part2[t3] = 3
    biais = Part1 + Part2
    
    # construction of the M series
    erreurs = list()
    muMat = np.empty((M, n), int)
    muMat[:] = -1
    tauMat = np.empty((M, K), int)

    # errors: a matrix (M x n) for each possible value of variance, 
    # giving the errors of each series at each position
    for i in np.arange(len(varianceError)):
        errorMat = np.empty((M, n))
        errorMat[:] = np.NaN
        for m in np.arange(M):
            errorMat[m,] = np.random.normal(0, np.sqrt(varianceError[i]), n)   
        erreurs.append(errorMat)
  
    for m in np.arange(M):
        # positions of breakpoints pour series m
        cond = 0
        while (cond == 0):
            # generar un numero aleatoriao uniforme entre
            # los n posibles puntos excluyendo el ultimo
            tauTmp = np.ceil((n-1)*np.random.uniform(0,1,K-1))
            tauTmp = np.sort(tauTmp)
            tauTmp = np.concatenate([tauTmp,[n]]).astype(int).copy()
            cond2 = 1
            
            for i in np.arange(1,K):
                cond2 = cond2 * ((tauTmp[i]-tauTmp[i-1])>=p2)
                
            for i in np.arange(K):
                for j in np.arange(len(pHaar)):
                    cond2 = cond2 * np.abs((tauTmp[i]-pHaar[j]))>=p1
            cond = cond2
        tauMat[m, np.arange(K)] = tauTmp
        
        # values of segments (segmentation part) for series m
        mutemp = np.random.choice(muChoix, size=1)
        muMat[m, np.arange(tauMat[m,0])] = np.repeat(mutemp, tauMat[m,0])
        muChoixTemp = muChoix
        
        if K > 1:
            for k in np.arange(1, K):
                toremove = np.nonzero(muChoixTemp == mutemp)
                muChoixTemp = np.delete(muChoixTemp, toremove).copy()
                mutemp = np.random.choice(muChoixTemp, size=1)

                muMat[m,np.arange(tauMat[m,k-1], tauMat[m,k])] = (
                      np.repeat(mutemp, (tauMat[m, k]-tauMat[m, k-1]))
                      )
    
    # final output as a data.frame se comenta ya que no se usa
    # biaisRep = biais.copy()
    
    #for i in np.arange(1,M):
    #    biaisRep = np.concatenate([biasisRep, biasis])
    
    series = np.array([], int)
    for i in np.arange(M):
        series = np.concatenate([series, np.repeat(i, n)])
    
    position = np.repeat(np.arange(n), M)
    mu = (muMat.T).copy().reshape((M*n,1))
    errors = erreurs[0].T.copy().reshape((M*n, 1))
    nomcolonnes = "erreur1"
  
  
    if len(varianceError) > 1:
        for i in np.arange(1,len(varianceError)):
            errors = pd.concat(errors, erreurs[i].T.copy().reshape(M*n, 1))
            nomcolonnes = np.concatenate([[nomcolonnes], "erreur"+str(i)])
        
    tau = np.array([], int)
    for m in np.arange(M):
        tauPos = np.repeat(0, n)
        tauPos[tauMat[m,]-1] = 1
        tau = np.concatenate([tau,tauPos])

    # Create a zipped list of tuples from above lists
    zippedList =  list(zip(series, position, mu.reshape(n), tau, biais, 
                       errors.reshape(n)))

    Profile = pd.DataFrame(zippedList, columns = ['series' , 'position', 'mu', 
                                                  'tau', 'biais', nomcolonnes])
    
    # output
    return(list([[K],muMat,tauMat,errorMat,biais,Profile]))

