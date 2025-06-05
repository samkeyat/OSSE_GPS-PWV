## The following script is to calculate Fraction Skill Score
## it can be used for any 2 dimensional dataset


import numpy as np
import random
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from netCDF4 import Dataset,num2date
import time
import glob

## 3 by 3 neighborhood evaluation ###########################################
def F_calc_FSS_NV9(m_RAIN_X,m_OBSV_X,threshold):
    shape = m_RAIN_X.shape
    nlat = shape[0]; nlon = shape[1]
    stat = np.zeros(shape=(nlat,nlon))  # create multi-dim array

    statR3 = []; statO3 = []
    for j in range(1,nlat-1):
        for i in range(1,nlon-1):
            count1 = 0
            count2 = 0

            statR = []; statO = []
            for jc in range(-1,2):
                for ic in range(-1,2):
                    if m_RAIN_X[j+jc,i+ic] >= threshold:
                        count1 += 1
                    if m_OBSV_X[j+jc,i+ic] >= threshold:
                        count2 += 1
                    statR.append(count1/9.0)
                    statO.append(count2/9.0)
            statRR = np.asarray(statR)
            statOO = np.asarray(statO)

            statR3.append(statRR)
            statO3.append(statOO)
    statR4 = np.asarray(statR3)
    statO4 = np.asarray(statO3)

    square = (statO4 - statR4)**2
    sumtotal = np.nansum(square.ravel())
    N = len(square.ravel())
    FBS = sumtotal / N

    statR_square = statR4**2
    statO_square = statO4**2
    sum_statR_sq = np.nansum(statR_square.ravel()) / N
    sum_statO_sq = np.nansum(statO_square.ravel()) / N
    FBS_ref = sum_statO_sq + sum_statR_sq

    FSS = 1 - (FBS / FBS_ref)
    FSS_fin = np.nanmean(FSS)
        
            
    return FSS_fin

## 5 by 5 neighborhood evaluation ############################################
def F_calc_FSS_NV25(m_RAIN_X,m_OBSV_X,threshold):
    shape = m_RAIN_X.shape
    nlat = shape[0]; nlon = shape[1]
    stat = np.zeros(shape=(nlat,nlon))  # create multi-dim array

    statR3 = []; statO3 = []
    for j in range(2,nlat-2):
        for i in range(2,nlon-2):
            count1 = 0
            count2 = 0

            statR = []; statO = []
            for jc in range(-2,3):
                for ic in range(-2,3):
                    if m_RAIN_X[j+jc,i+ic] >= threshold:
                        count1 += 1
                    if m_OBSV_X[j+jc,i+ic] >= threshold:
                        count2 += 1
                    statR.append(count1/25.0)
                    statO.append(count2/25.0)
            statRR = np.asarray(statR)
            statOO = np.asarray(statO)

            statR3.append(statRR)
            statO3.append(statOO)
    statR4 = np.asarray(statR3)
    statO4 = np.asarray(statO3)

    square = (statO4 - statR4)**2
    sumtotal = np.nansum(square.ravel())
    N = len(square.ravel())
    FBS = sumtotal / N

    statR_square = statR4**2
    statO_square = statO4**2
    sum_statR_sq = np.nansum(statR_square.ravel()) / N
    sum_statO_sq = np.nansum(statO_square.ravel()) / N
    FBS_ref = sum_statO_sq + sum_statR_sq

    FSS = 1 - (FBS / FBS_ref)
    FSS_fin = np.nanmean(FSS)
        
            
    return FSS_fin