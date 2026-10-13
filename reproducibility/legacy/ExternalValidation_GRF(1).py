# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 08:07:30 2024

@author: hulle
"""

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
# def load_excel_files(folder_path):
input_folder_path = 'Input/ng RightRANK.xlsx'
    
baseline_folder = 'Baseline/ng RightRNormalisedGRF.xlsx'
# input_folder_path = 'Input/ng LeftLANK.xlsx'
    
# baseline_folder = 'Baseline/ng LeftLNormalisedGRF.xlsx'

# model_directory = 'D:/Predict Kinetics/Results/ANKLE-GRF/Long/Left/PINN/1D Convo/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Left/PINN/LSTM/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Left/PINN/GRU/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Right/PINN/1D Convo/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Right/PINN/LSTM/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Right/PINN/GRU/Model/'

# model_directory = 'D:/Predict Kinetics/Results/ANKLE-GRF/Long/Left/SEQ2SEQ/1DConvo-LSTM/Model/'
# model_directory = 'D:/Predict Kinetics/Results/ANKLE-GRF/Long/Left/SEQ2SEQ/LSTM/Model/'
# model_directory = 'D:/Predict Kinetics/Results/ANKLE-GRF/Long/Left/SEQ2SEQ/GRU/Model/'
model_directory = 'D:/Predict Kinetics/Results/ANKLE-GRF/Long/Right/SEQ2SEQ/1DConvo-LSTM/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Right/SEQ2SEQ/LSTM/Model/'
# model_directory = 'E:/Predict Kinetics/Results/ANKLE-GRF/Long/Right/SEQ2SEQ/GRU/Model/'


h5_files = [f for f in os.listdir(model_directory) if f.endswith('.h5')]

#%%
for model_name in h5_files:
    # model_name = 'AP LeftLANK10_epoch_.h5'
    model = load_model(os.path.join(model_directory, model_name), compile=False)
    
    input_shape = model.input_shape
    
    print(os.path.join(model_directory, model_name))
    
    input_data = pd.read_excel(input_folder_path).dropna()
    baseline_data = pd.read_excel(baseline_folder)
    
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Left/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Left/1D-Convo/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Left/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Left/LSTM/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Left/GRU/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Left/GRU/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/GRU/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Right/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Right/1D-Convo/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Right/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Right/LSTM/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long RIght/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Right/GRU/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Right/GRU/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/GRU/'+ model_name[:-3], exist_ok=True)
    

    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Left/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Left/1D-Convo/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Left/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Left/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Left/LSTM/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Left/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Left/GRU/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Left/GRU/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Left/GRU/'+ model_name[:-3], exist_ok=True)
   
    os.makedirs('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Right/1D-Convo/'+ model_name[:-3], exist_ok=True)
    os.makedirs('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Right/1D-Convo/' +  model_name[:-3], exist_ok=True)
    os.makedirs('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Right/1D-Convo/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Right/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Right/LSTM/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long RIght/LSTM/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Right/GRU/'+ model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Right/GRU/' +  model_name[:-3], exist_ok=True)
    # os.makedirs('E:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Right/GRU/'+ model_name[:-3], exist_ok=True)
        

        
    results = []
    i = 0
    for col in input_data.columns:
        # col = 'SUBJ1 (3_AP_Left_cycles'
        # print(col)
        input_df = input_data[col]
        input_df = input_df.reindex(range(input_shape[1])).fillna(0)
        input_df = pd.DataFrame([input_df]).transpose()
        samples_in = input_df.shape[0]
        timesteps_in = 1
        features_in = 1
        sequence_in = input_df.values.reshape((timesteps_in,samples_in,  features_in))
        # sequence_in = input_df.values.reshape(input_shape)
        # num_samples = len(input_df)
        # reshaped_data = np.zeros((input_shape[1:]))
        
        
    
        # input_data = input_df.values
        # reshaped_data[:input_data.shape[0], :input_data.shape[1]] = input_data
        
        # print(sequence_in.shape)
        
        
        baseline_df = baseline_data[col]
        baseline_df = baseline_df.reindex(range(input_shape[1])).fillna(0)
        
        predictions = model.predict(sequence_in, verbose=0)
        predictions = predictions.flatten()  # Ensure predictions is a 1D array
        
        rmse = np.sqrt(mean_squared_error(baseline_df, predictions))
        nrmse = rmse / (baseline_df.max() - baseline_df.min())
        mae = mean_absolute_error(baseline_df, predictions)
        stat, p_pred = shapiro(predictions)
        stat, p_base = shapiro(baseline_df)
        
        if p_pred > 0.05 and p_base > 0.05:
            stat, p = ttest_ind(predictions, baseline_df)
        else:
            stat, p = mannwhitneyu(predictions, baseline_df)
        
        results.append({
            # 'File': col,
            'Column': col,
            'RMSE': rmse,
            'nRMSE': nrmse,
            'MAE': mae,
            'p-value': p,
            'Normality (Predictions)': p_pred,
            'Normality (Baseline)': p_base
        })
        predictions_df = pd.DataFrame([predictions]).transpose()      
        out = pd.concat([pd.DataFrame(predictions_df[0]), baseline_df], axis=1)
        
        # ind = k
        x = np.linspace(0,input_shape[1]-1,input_shape[1])
        # x_in = np.
        y_pred = predictions_df[0]
        y_true = baseline_df
        # plt.plot(x,x_in,label ="Ankle Position")
        plt.plot(x,y_pred,label ="Predicted GRF (pred)")
        plt.plot(x,y_true,label ="Measured GRF (expt)")
        plt.legend()
        # plt.savefig('D:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Left/1D-Convo/'+ model_name[:-3] + '/' +'PINN' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Left/LSTM/'+ model_name[:-3] + '/' +'PINN' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Left/GRU/'+ model_name[:-3] + '/' +'PINN' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Right/1D-Convo/'+ model_name[:-3] + '/' +'PINN' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Right/LSTM/'+ model_name[:-3] + '/' +'PINN' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/PINN/Long Right/GRU/'+ model_name[:-3] + '/' +'PINN' + str(i) + '.png')
        #
        # plt.savefig('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Left/1D-Convo/'+ model_name[:-3] + '/' +'SEQ2SEQ' + str(i) + '.png')
        # plt.savefig('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Left/LSTM/'+ model_name[:-3] + '/' +'SEQ2SEQ' + str(i) + '.png')
        # plt.savefig('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Left/GRU/'+ model_name[:-3] + '/' +'SEQ2SEQ' + str(i) + '.png')
        plt.savefig('D:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Right/1D-Convo/'+ model_name[:-3] + '/' +'seq2seq' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Right/LSTM/'+ model_name[:-3] + '/' +'SEQ2SEQ' + str(i) + '.png')
        # plt.savefig('E:/Predict Kinetics/External Validation GRF/Figure/SEQ2SEQ/Long Right/GRU/'+ model_name[:-3] + '/' +'SEQ2SEQ' + str(i) + '.png')
        
        
        # plt.figure()
        plt.close()
        i = i+1
        
        results_df = pd.DataFrame(results)
        # results_df.to_excel('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Left/1D-Convo/' +  model_name[:-3] + '/'+'Long LeftLNormalisedGRF'+ 'PINN'+str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/1D-Convo/'+ model_name[:-3] +'/' +'Long LeftLNormalisedGRF' +'PINN'+str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Left/LSTM/' +  model_name[:-3] + '/'+'Long LeftLNormalisedGRF'+ 'PINN' +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/LSTM/'+ model_name[:-3] +'/' +'Long LeftLNormalisedGRF' +'PINN' +str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Left/GRU/' +  model_name[:-3] + '/'+'Long LeftLNormalisedGRF'+ 'PINN' +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/GRU/'+ model_name[:-3] +'/' +'Long LeftLNormalisedGRF' +'PINN' +str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Right/1D-Convo/' +  model_name[:-3] + '/'+ 'PINN' +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/1D-Convo/'+ model_name[:-3] +'/' +'PINN' +str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Right/LSTM/' +  model_name[:-3] + '/'+'Long RightRNormalisedGRF'+ 'PINN'  +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/LSTM/'+ model_name[:-3] +'/'+'Long RightRNormalisedGRF' +'PINN' +str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/PINN/Long Right/GRU/' +  model_name[:-3] + '/'+'Long RightRNormalisedGRF'+ 'PINN' + str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/GRU/'+ model_name[:-3] +'/' +'Long RightRNormalisedGRF' +'PINN' +str(i) + '.xlsx' )
        ##'
        # results_df.to_excel('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Left/1D-Convo/' +  model_name[:-3] + '/'+'Long LeftLNormalisedGRF'+ 'SEQ2SEQ'+str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Left/1D-Convo/'+ model_name[:-3] +'/' +'Long LeftLNormalisedGRF' +'SEQ2SEQ' +str(i)+ '.xlsx' )
        # results_df.to_excel('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Left/LSTM/' +  model_name[:-3] + '/'+'Long LeftLNormalisedGRF'+ 'SEQ2SEQ' +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Left/LSTM/'+ model_name[:-3] +'/' +'Long LeftLNormalisedGRF' +'SEQ2SEQ' +str(i) + '.xlsx' )
        # results_df.to_excel('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Left/GRU/' +  model_name[:-3] + '/'+'Long LeftLNormalisedGRF'+ 'SEQ2SEQ' +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Left/GRU/'+ model_name[:-3] +'/' +'Long LeftLNormalisedGRF' +'SEQ2SEQ' +str(i) + '.xlsx' )
        results_df.to_excel('D:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Right/1D-Convo/' +  model_name[:-3] + '/'+ 'SEQ2SEQ' +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        out.to_excel('D:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Right/1D-Convo/'+ model_name[:-3] +'/' +'SEQ2SEQ' +str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Right/LSTM/' +  model_name[:-3] + '/'+'Long RightRNormalisedGRF'+ 'SEQ2SEQ'  +str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Right/LSTM/'+ model_name[:-3] +'/'+'Long RightRNormalisedGRF' +'SEQ2SEQ' +str(i) + '.xlsx' )
        # results_df.to_excel('E:/Predict Kinetics/External Validation GRF/Statistical Analysis/SEQ2SEQ/Long Right/GRU/' +  model_name[:-3] + '/'+'Long RightRNormalisedGRF'+ 'SEQ2SEQ' + str(i) + '.xlsx', sheet_name= 'Stats',index=False)
        # out.to_excel('E:/Predict Kinetics/External Validation GRF/Predictions/SEQ2SEQ/Long Right/GRU/'+ model_name[:-3] +'/' +'Long RightRNormalisedGRF' +'SEQ2SEQ' +str(i) + '.xlsx' )
        
        

