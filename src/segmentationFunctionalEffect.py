# -*- coding: utf-8 -*-
"""
Create Fri Aug 07 17:23:45

@author: nnarria
"""

import numpy as np
from scipy.stats import invgamma



#%%
##############################################################################
# estimation of the breakpoints and of the functions composing
# the functional part (Metropolis-Hastings algorithm)
##############################################################################

# INPUTS
# serie: the observations
# nbiter: number of MH iterations total (burn-in + post burn-in)
# nburn: number of burn-in iterations
# lec1: chosen value for parameter c1
# lec2: chosen value for parameter c2
# Fmatrix: matrix which gives the values of the functions in the dictionary, 
# at each position.
# nbSegInit: initial number of segments for the segmentation part

# nbToChangegamma: number of gamma components proposed to be changed at each 
# iteration (when we propose to modify gamma), that is the number of inclusion 
# or deletion of breakpoints

# nbFuncInit: initial number of functions from the dictionary for the 
# functional part
# nbToChanger: number of r components proposed to be changed at each 
# iteration (when we propose to modify r), that is the number of inclusion or 
# deletion of functions from the dictionary
# Pi: vector with prior probabilities for the breakpoints: for position l 
# it is the prior probability that we observe a difference between the 
# segmentation part at time l and the segmentation part at time (l-1). 
# By convention Pi[1] = 1.
# eta: vector with prior probabilities for the functions in the dictionary: 
# eta[j] gives the prior proba that the function j from the dictionary will 
# be included in the functional part. By convention eta[1] = 1.
# printiter: if TRUE, the number of the actual iteration is plotted

# OUTPUTS
# sumgamma: for the breakpoints, for component l: number of iterations during 
# which a difference between the segmentation part at time l and the 
# segmentation part at time (l-1) was non nul (during post-burn-in).
# sumr: for functions from the dictionary, for component l: number of 
# iterations during which the function j from the dictionary was included in 
# the functional part (during post-burn-in).
# nbactugamma: number of iterations during which a MH proposal for the 
# breakpoints has been accepted (during which gamma has been modified) 
# (among iterations during burn-in and post-burn-in).
# nbactur: number of iterations during which a MH proposal for the functions 
# from the dictionary has been accepted (during which r has been modified) 
# (among iterations during burn-in and post-burn-in).
# gammamatrix: matrix to store all the post-burn-in simulations for gamma: 
# one line corresponds to one simulation (iteration)
# rmatrix: matrix to store all the post-burn-in simulations for gamma r: one 
# line corresponds to one simulation (iteration)

# serie tiene que ser un arreglo de nx1
def segmentation_bias_MH(serie, nbiter, nburn, lec1, lec2, Fmatrix, nbSegInit,
                         nbToChangegamma, nbFuncInit, nbToChanger, Pi, eta,
                         printiter=True):
    
    # de matrix to vector
    y = serie.reshape(-1)
    n = len(y)
    X = np.tri(n, n, 0, dtype=int)
    J = Fmatrix.shape[1]

    # to store results    
    gammamatrix = np.empty((nbiter-nburn, n))
    gammamatrix.fill(np.nan)
    rmatrix = np.empty((nbiter-nburn,J))
    rmatrix.fill(np.nan)
    
    sumgamma = np.zeros(n, int) 
    nbactugamma = 0
    sumr = np.zeros(J, int)
    nbactur = 0
    
    
    # initialization gamma 
    indgamma10 = np.random.choice(np.arange(1, n), size=nbSegInit-1, 
                                  replace=False)    
    gamma0 = np.zeros(n, int)
    gamma0[0] = 1
    
    # asigna desde 0 a (nbSegInit-1)-1
    for i in np.arange(nbSegInit-1):
        gamma0[indgamma10[i]] = 1

    indgamma1 = np.concatenate((np.array([0]), indgamma10))
    gamma = gamma0.copy()
    nbSeg = nbSegInit
    Xgamma = X[:,indgamma1] # fancy indexing
    invUgamma = np.diag(np.ones(n)) - lec1/(1+lec1) * (
        Xgamma @ np.linalg.inv(Xgamma.T @ Xgamma) @ Xgamma.T)
    
        
    # initialization r
    indr10 = np.random.choice(np.arange(1, J), size=nbFuncInit-1, 
                                  replace=False)
    
    r0 = np.zeros(J, int)
    r0[0] = 1
    for i in np.arange(nbFuncInit-1):
        r0[indr10[i]] = 1

    indr1 = np.concatenate((np.array([0]), indr10))
    r = r0
    nbFunc = nbFuncInit
    Fmatrixr = Fmatrix[:,indr1]

        
    temp1 = np.linalg.det(
        np.linalg.inv(
            Fmatrixr.T @ (invUgamma + np.identity(n)/lec2) @ Fmatrixr
            )
        )
    
    temp2 = y.T @ (
        invUgamma - invUgamma @ Fmatrixr @ np.linalg.inv(
                Fmatrixr.T @  (invUgamma + np.identity(n)/lec2) @ Fmatrixr
            ) @ Fmatrixr.T @ invUgamma) @ y

    temp3 = np.linalg.det(np.dot(Fmatrixr.T, Fmatrixr))
    

    # iterations MH
    for iter in np.arange(nbiter):        
        if printiter == True:
            print("iter " + str(iter+1))
        
        # uniform choice between several movements
        choix = np.random.choice([1,2], size=1)
        
        # movement proposing to change only gamma
        if choix == 1:
            gammaprop = gamma.copy()
            indgamma1prop = indgamma1.copy()
            nbSegprop = nbSeg
            indToChange = np.random.choice(np.arange(1,n), 
                                           size=nbToChangegamma,
                                           replace=False)
            
            for i in np.arange(nbToChangegamma):
                if gamma[indToChange[i]] == 0:
                    gammaprop[indToChange[i]] = 1
                    indgamma1prop = np.concatenate([indgamma1prop, 
                                                   [indToChange[i]]])                   
                    nbSegprop = nbSegprop + 1
                else:
                    gammaprop[indToChange[i]] = 0                    
                    indremove = np.nonzero(indgamma1prop==indToChange[i])[0]
                    indgamma1prop = np.delete(indgamma1prop, indremove[0])          
                    nbSegprop = nbSegprop - 1
            
            Xgammaprop = X[:,indgamma1prop]
 
            invUgammaprop = np.identity(n) - (lec1/(1+lec1) * (
                Xgammaprop @ np.linalg.inv(Xgammaprop.T @ Xgammaprop) @ 
                Xgammaprop.T))

            # new nnarria
            tmpsolve_ = np.linalg.inv(
                Fmatrixr.T @ (invUgammaprop + np.identity(n)/lec2) @ Fmatrixr)
           
            temp1prop = np.linalg.det(tmpsolve_)
            temp2prop = y.T @ (invUgammaprop - invUgammaprop @ Fmatrixr @ (
                    tmpsolve_
                ) @ Fmatrixr.T @ invUgammaprop
            ) @ y
    
            A = np.power(1+lec1, (nbSeg-nbSegprop)/2) * np.prod(
                    np.power(Pi[1:]/(1-Pi[1:]),(gammaprop-gamma)[1:])
                ) * np.power(temp1prop/temp1, 1/2) * (
                np.float_power(temp2/temp2prop, n/2))
            
            probaccept1 = min(1, A)
            seuil = np.random.uniform(0,1,1)[0] # runif(1)
            
            if seuil < probaccept1:
                gamma = gammaprop.copy()                
                indgamma1 = indgamma1prop.copy()
                nbSeg = nbSegprop
                Xgamma = Xgammaprop.copy()
                invUgamma = invUgammaprop.copy()
                temp1 = temp1prop
                temp2 = temp2prop.copy()
                nbactugamma = nbactugamma + 1
            
            
        # movement proposing to change only r
        if choix == 2:
            rprop = r.copy()
            indr1prop = indr1.copy()
            nbFuncprop = nbFunc
            indToChange = np.random.choice(np.arange(1,J), 
                                           size=nbToChanger,
                                           replace=False)

            for i in np.arange(nbToChanger):
                if r[indToChange[i]] == 0:
                    rprop[indToChange[i]] = 1
                    indr1prop = np.concatenate([indr1prop, [indToChange[i]]])
                    nbFuncprop = nbFuncprop + 1
                else:
                    rprop[indToChange[i]] = 0
                    indremove = np.nonzero(indr1prop == indToChange[i])[0]
                    indr1prop = np.delete(indr1prop, indremove[0])
                    nbFuncprop = nbFuncprop - 1

            Fmatrixrprop = Fmatrix[:, indr1prop]
            
            # new nnarria
            tmpsolve_ = np.linalg.inv(Fmatrixrprop.T @ (
                    invUgamma + np.identity(n)/lec2) @ Fmatrixrprop)
            
            temp1prop = np.linalg.det(tmpsolve_)
            temp2prop = y.T @ (invUgamma - invUgamma @ Fmatrixrprop @  
                               tmpsolve_ @ Fmatrixrprop.T @ invUgamma) @ y
            temp3prop = np.linalg.det(Fmatrixrprop.T @ Fmatrixrprop)


            A = np.power(lec2, (nbFunc-nbFuncprop)/2) * np.prod(
                    np.power(eta[1:]/(1-eta[1:]),(rprop-r)[1:])
                ) * np.power(temp1prop/temp1, 1/2) * (
                    np.float_power(temp2/temp2prop, n/2)) * ( 
                        np.power(temp3prop/temp3, 1/2))
                         
                        
            probaccept1 = min(1, A)
            seuil = np.random.uniform(0,1,1)[0] # runif(1)   
            
            if seuil < probaccept1:
                r = rprop.copy()
                indr1 = indr1prop.copy()
                nbFunc = nbFuncprop
                Fmatrixr = Fmatrixrprop.copy()
                temp1 = temp1prop
                temp2 = temp2prop.copy()
                temp3 = temp3prop
                nbactur = nbactur + 1
        
        # store results when we are in post-burn-in
        if iter >= nburn:
            sumgamma = sumgamma + gamma
            sumr = sumr + r
            gammamatrix[(iter-nburn),:] = gamma.copy()
            rmatrix[(iter-nburn),:] = r.copy()
    
    # devuelve un diccionario
    return dict(sumgamma=sumgamma, sumr=sumr, nbactugamma=nbactugamma, 
            nbactur=nbactur, gammamatrix=gammamatrix, rmatrix=rmatrix)


#%% 
##############################################################################
# After the segmentation: 
# Estimation of betagamma, 
# lambdar and sigma2 (Gibbs sampler)   
##############################################################################

# INPUTS
# serie: the observations
# nbiter: total number of iterations for the Gibbs sampler (burn-in + 
# post burn-in)
# nburn: number of burn-in iterations
# lec1: chosen value for the c1 parameter
# lec2: chosen value for the c2 parameter
# Fmatrix: matrix which gives the values of the functions in the dictionary, 
# at each position.
# gammahat: estimated gamma vector (using the preceding MH algo) 
# rhat: estimated r vector (using the preceding MH algo) 
# priorminsigma2: prior minimum value for sigma2
# priormaxsigma2: prior maximum value for sigma2
# printiter: if TRUE, the number of the actual iteration is plotted

# OUTPUTS
# resbetagammahat: matrix to store all the post-burn-in simulations for 
# betagamma: one line corresponds to one simulation (iteration)
# reslambdarhat: matrix to store all the post-burn-in simulations for 
# lambdar: one line corresponds to one simulation (iteration)
# ressigma2hat: matrix to store all the post-burn-in simulations for sigma2: 
# one line corresponds to one simulation (iteration)
# estbetagamma: estimation of betagamma (mean of all the simulated betagamma)
# estlambdar: estimation of lambdar (mean of all the simulated lambdar)
# estsigma2: estimation of sigma2 (mean of all the simulated sigma2)



def estimation_moy_biais (serie, nbiter, nburn, lec1, lec2, Fmatrix, gammahat,
                          rhat, priorminsigma2, priormaxsigma2, 
                          printiter=True):
    y = serie[0:,0]
    n = len(y)
    X = np.tri(n, n, 0, dtype=int)    
   
    dgammahat = np.sum(gammahat)
    drhat = np.sum(rhat)
   
    if drhat != 0:
        
        ### 
        directions_x3 = np.array([[ 0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0],
                       [2, 0, 1], [2, 1, 0]])
        matrix_dir_x3 = np.random.choice(np.arange(directions_x3.shape[0]), 
                                                   size=nbiter)
        ###
        
        # useful quantities for calculations
        Xgamma = X[:,np.nonzero(gammahat==1)[0]]
        Fmatrixr = Fmatrix[:,np.nonzero(rhat==1)[0]]
        temp1 = np.linalg.inv((1+lec1)/lec1 * Xgamma.T @ Xgamma)
        temp2 = np.linalg.inv((1+lec2)/lec2 * Fmatrixr.T @ Fmatrixr)
        alpha = (n+dgammahat+drhat)/2
        temp3 = Xgamma.T @ Xgamma/lec1
        temp4 = Fmatrixr.T @ Fmatrixr/lec2
       
        # to store results     
        resbetagammahat = np.zeros(dgammahat*(nbiter-nburn)).reshape(
            dgammahat, -1)       
        reslambdarhat = np.zeros(int(drhat*(nbiter-nburn))).reshape(
            int(drhat), -1)
        ressigma2hat = np.zeros(nbiter-nburn)

       
        # initializations
        sigma2hat = np.exp(np.random.uniform(np.log(priorminsigma2), 
                                          np.log(priormaxsigma2), 1))[0]
         
        mean_ = np.zeros(dgammahat)
        cov_ = lec1*sigma2hat*np.linalg.inv(Xgamma.T @ Xgamma)
        betagammahat = np.random.multivariate_normal(mean_, cov_, 1)[0]
      
        mean_ = np.zeros(int(drhat))
        cov_ = lec2*sigma2hat*np.linalg.inv(Fmatrixr.T @ Fmatrixr)
        lambdarhat = np.random.multivariate_normal(mean_, cov_, 1)[0]
        
        # iterations
        for iter in np.arange(nbiter):
            if printiter == True:
                print('iter ' + str(iter))
            
            ordre = directions_x3[matrix_dir_x3[iter]]
            
            for j in np.arange(3):
                if ordre[j] == 0:
                    mean_ = temp1 @ Xgamma.T @ (y-Fmatrixr @ lambdarhat)
                    cov_ = sigma2hat*temp1
                    betagammahat = np.random.multivariate_normal(
                        mean_, cov_, 1)[0]
                
                if ordre[j] == 1:
                    mean_ = temp2 @ Fmatrixr.T @ (y-Xgamma @ betagammahat)
                    cov_ = sigma2hat*temp2
                    lambdarhat = np.random.multivariate_normal(
                        mean_, cov_, 1)[0]
                    
                if ordre[j] == 2:
                    scale_ = 1/2 * (
                        (y-Xgamma @ betagammahat-Fmatrixr @ lambdarhat).T @ (
                            y-Xgamma @ betagammahat-Fmatrixr @ lambdarhat
                            ) + betagammahat.T @ temp3 @ betagammahat +
                            lambdarhat.T @ temp4 @ lambdarhat)
                    sigma2hat = invgamma.rvs(alpha, scale=scale_, size=1)
            
            if iter >= nburn:
                resbetagammahat[:, iter-nburn] = betagammahat
                reslambdarhat[:, iter-nburn] = lambdarhat
                ressigma2hat[iter-nburn] = sigma2hat
        
        
        estbetagamma = np.sum(resbetagammahat, axis=1)/(nbiter-nburn)
        estlambdar = np.sum(reslambdarhat, axis=1)/(nbiter-nburn)
        estsigma2 = np.sum(ressigma2hat)/(nbiter-nburn)
            
    if drhat == 0:
        
        ###
        directions_x2 = np.array([[ 0, 1],[1, 0]])
        matrix_dir_x2 = np.random.choice(np.arange(directions_x2.shape[0]), 
                                                   size=nbiter)
        ###
        
        # useful quantities for calculations
        Xgamma = X[:,np.nonzero(gammahat==1)[0]]
        temp1 = np.linalg.inv((1+lec1)/lec1 * Xgamma.T @ Xgamma)
        alpha = (n+dgammahat)/2
        temp3 = Xgamma.T @ Xgamma/lec1

        # to store results
        resbetagammahat = np.zeros(dgammahat*(nbiter-nburn)).reshape(
            dgammahat, -1)
        reslambdarhat = None
        ressigma2hat = np.zeros(nbiter-nburn)
        
        # initializations
        sigma2hat = np.exp(np.random.uniform(np.log(priorminsigma2), 
                                          np.log(priormaxsigma2), 1))[0]
        mean_ = np.zeros(dgammahat)
        cov_ = lec1*sigma2hat*np.linalg.inv(Xgamma.T @ Xgamma)
        betagammahat = np.random.multivariate_normal(mean_, cov_, 1)[0]
     
        # iterations
        for iter in np.arange(nbiter):
            if printiter == True:
                print("iter " + str(iter))
            
            ordre = directions_x2[matrix_dir_x2[iter]]
            
            for j in np.arange(2):
                if ordre[j] == 0:
                    mean_ = temp1 @ Xgamma.T @ y
                    cov_ = sigma2hat*temp1
                    betagammahat = np.random.multivariate_normal(
                        mean_, cov_, 1)[0]
                
                if ordre[j] == 1:                    
                    scale_ = 1/2 * ((y-Xgamma @ betagammahat).T @ (
                            y-Xgamma @ betagammahat) + betagammahat.T @ (
                                temp3 @ betagammahat))
                    sigma2hat = invgamma.rvs(alpha, scale=scale_, size=1)
            
            if iter >= nburn:
                resbetagammahat[:, iter-nburn] = betagammahat
                ressigma2hat[iter-nburn] = sigma2hat
        
        estbetagamma = np.sum(resbetagammahat, axis=1)/(nbiter-nburn)
        estlambdar = None
        estsigma2 = np.sum(ressigma2hat)/(nbiter-nburn)     


    return list([resbetagammahat, reslambdarhat, ressigma2hat, 
                           estbetagamma, estlambdar, estsigma2])



# %%


##############################################################################
# AThe two steps of M-H and Gibbs sampler are integrated for the 
# estimation of points of change with functional effect 
##############################################################################

# INPUTS
# Fmatrix: Matrix with the dictionary data
# data_serie:data series to which you want to detect change points
# itertot: Total number of iterations to be used in both M-H and Gibbs Sampler
# burnin: Number of iterations that will not be considered
# lec1: chosen value for the c1 parameter
# lec2: chosen value for the c2 parameter
# nbseginit: initial number of segments for the segmentation part
# nfuncinit: initial number of functions from the dictionary for the 
# functional part
# nbtochangegamma: number of gamma components proposed to be changed at each 
# iteration (when we propose to modify gamma), that is the number of inclusion 
# or deletion of breakpoints
# nbtochanger: number of r components proposed to be changed at each 
# iteration (when we propose to modify r), that is the number of inclusion or 
# deletion of functions from the dictionary
# Pi: vector with prior probabilities for the breakpoints: for position l 
# it is the prior probability that we observe a difference between the 
# segmentation part at time l and the segmentation part at time (l-1). 
# By convention Pi[1] = 1.
# eta: vector with prior probabilities for the functions in the dictionary: 
# eta[j] gives the prior proba that the function j from the dictionary will 
# be included in the functional part. By convention eta[1] = 1.
# threshold_bp: probability threshold from which the change points 
# are selected
# threshold_fnc: probability threshold from which the functions are selected
# printiter: if TRUE, the number of the actual iteration is plotted

    
# OUTPUTS
# list with;
# [0]: list with
# sumgamma: for the breakpoints, for component l: number of iterations 
# during which a difference between the segmentation part at time l and the 
# segmentation part at time (l-1) was non nul (during post-burn-in).
# sumr: for functions from the dictionary, for component l: number of 
# iterations during which the function j from the dictionary was included 
# in the functional part (during post-burn-in).
# nbactugamma: number of iterations during which a MH proposal for the 
# breakpoints has been accepted (during which gamma has been modified) 
# (among iterations during burn-in and post-burn-in).
# nbactur: number of iterations during which a MH proposal for the functions 
# from the dictionary has been accepted (during which r has been modified) 
# (among iterations during burn-in and post-burn-in).
# gammamatrix: matrix to store all the post-burn-in simulations for gamma: 
# one line corresponds to one simulation (iteration).
# rmatrix: matrix to store all the post-burn-in simulations for gamma r: one 
# line corresponds to one simulation (iteration)
# [1] resbetagammahat: matrix to store all the post-burn-in simulations for.
# [2] reslambdarhat: matrix to store all the post-burn-in simulations for. 
# [3] ressigma2hat: matrix to store all the post-burn-in simulations for.
# [4] estbetagamma: estimation of betagamma (mean of all the simulated 
# betagamma)
# [5] estlambdar: estimation of lambdar (mean of all the simulated lambdar)
# [6] estsigma2: estimation of sigma2 (mean of all the simulated sigma2)
# [7] reconstructiontot

def dbp_with_function_effect (
        Fmatrix, data_serie, itertot, burnin, 
        lec1, lec2, nbseginit, nbfuncinit, nbtochangegamma, nbtochanger,
        Pi, eta, threshold_bp, threshold_fnc, printiter=False):
    
    n = len(data_serie)
    
    # result Metropolis Hastings
    resMH = segmentation_bias_MH (
        data_serie, itertot, burnin, lec1, 
        lec2, Fmatrix, nbseginit, 
        nbtochangegamma, nbfuncinit, 
        nbtochanger, Pi, eta, 
        printiter=printiter)
    print('resMH calculado')
    
    
    # Selection of the points of change
    breakpoints = np.asarray(
    np.nonzero(resMH['sumgamma']/(itertot-burnin) > threshold_bp))[0]
    print('breakpoints: ' + str(breakpoints))


    gammahat = np.zeros(n, int)
    for i in (breakpoints):
        gammahat[i] = 1  


    # Selection of the functions
    basefunctions = np.asarray(
        np.nonzero(resMH['sumr']/(itertot-burnin) > threshold_fnc))[0]
    print('basefunctions: ' + str(basefunctions))
    
    rhat = np.zeros(Fmatrix.shape[1])
    for i in basefunctions[1:]:
        rhat[i] = 1    
 
    
    # Estimation of betagamma, lambdar and sigma2
    # To calculate the hope of the series, the segmentation and 
    # functional part is estimated and then added.        
    priorminsigma2 = 0.001
    priormaxsigma2 = 5
        
    estim = estimation_moy_biais(
        data_serie, itertot, burnin, lec1, lec2, 
        Fmatrix, gammahat, rhat, priorminsigma2, 
        priormaxsigma2, printiter=printiter)
    
    muest = np.zeros(breakpoints.shape[0])
    muest[0] = estim[3][0]
    reconstructionmu = np.zeros(Fmatrix.shape[0])
    breakpoints_extend = np.concatenate([breakpoints, [Fmatrix.shape[0]]])
        
    if breakpoints.shape[0] > 1:
        muest = np.cumsum(estim[3])
                
        for i in np.arange(len(breakpoints_extend)-1):
            reconstructionmu[
                breakpoints_extend[i]:breakpoints_extend[i+1]] = muest[i]
        
    reconstructionf = np.zeros(Fmatrix.shape[0])
    if basefunctions.shape[0] > 1:
        for i in np.arange(1, basefunctions.shape[0]):
            reconstructionf = reconstructionf + estim[4][i-1] * (
                Fmatrix[:,basefunctions[i]])
        
    reconstructiontot = reconstructionmu + reconstructionf
    print("reconstruction tot OK")


    return list([resMH, estim[0], estim[1], estim[2], estim[3], estim[4], 
                 estim[5], reconstructiontot])
    
    
