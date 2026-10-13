# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:24:45 2023

@author: hulle
"""
#%%
import pandas as pd
from scipy.stats import shapiro, wilcoxon, ttest_ind, mannwhitneyu
import numpy as np

data = pd.read_excel('C:/Users/hulle/OneDrive/Desktop/2023/Journal/Undreprocess/4- iMPROVING POSE ESTIMATION/hp optimization/Analysis in one.xlsx', sheet_name='OriginPro_Sheet')


L_Ankle_fast_MSE = data.iloc[2:39, 3:7]
L_Ankle_fast_RMSE = data.iloc[2:39, 9:13]
L_Ankle_fast_nRMSE = data.iloc[2:39, 15:19]
L_Ankle_fast_MAE = data.iloc[2:39, 21:25]

# data_segregated = [str(L_Ankle_fast_MSE),str(L_Ankle_fast_RMSE),str(L_Ankle_fast_nRMSE),str(L_Ankle_fast_MAE)]


stat1, p_value1 = shapiro(L_Ankle_fast_MAE.iloc[:,0].values)
stat2, p_value2 = shapiro(L_Ankle_fast_MAE.iloc[:,1].values)
stat3, p_value3 = shapiro(L_Ankle_fast_MAE.iloc[:,2].values)
stat4, p_value4 = shapiro(L_Ankle_fast_MAE.iloc[:,3].values)
alpha = 0.05

if p_value1> alpha and p_value2>alpha and p_value3>alpha and p_value4>alpha:
    ## normal distribution rejecting null hypothesis
    print('normal')
    # t_stat_LSTM, t_p_value_LSTM     = ttest_ind(np.array([L_Ankle_fast_MAE.iloc[:,0].values], dtype = float), np.array([L_Ankle_fast_MAE.iloc[:,1].values], dtype = float))
    # t_stat_1D, t_p_value_1D         = ttest_ind(L_Ankle_fast_MAE.iloc[:,0].values, L_Ankle_fast_MAE.iloc[:,2].values)
    # t_stat_GRU, t_p_value_GRU       = ttest_ind(L_Ankle_fast_MAE.iloc[:,0].values, L_Ankle_fast_MAE.iloc[:,3].values)
    # t_p_value = pd.DataFrame([t_p_value_LSTM, t_p_value_1D,t_p_value_GRU])
    # t_p_value.to_excel('L_Ankle_fast_t', sheet_name='MAE')
else:
    ## non normal distribution
    print('non normal')
    # u_stat_LSTM, u_p_value_LSTM     = mannwhitneyu(L_Ankle_fast_MAE.iloc[:,0].values, L_Ankle_fast_MAE.iloc[:,1].values)
    # u_stat_1D, u_p_value_1D         = mannwhitneyu(L_Ankle_fast_MAE.iloc[:,0].values, L_Ankle_fast_MAE.iloc[:,2].values)
    # u_stat_GRU, u_p_value_GRU       = mannwhitneyu(L_Ankle_fast_MAE.iloc[:,0].values, L_Ankle_fast_MAE.iloc[:,3].values)
    # U_p_value = pd.DataFrame([u_p_value_LSTM, u_p_value_1D,u_p_value_GRU])
    # U_p_value.to_excel('L_Ankle_fast_u', sheet_name='MAE')
    