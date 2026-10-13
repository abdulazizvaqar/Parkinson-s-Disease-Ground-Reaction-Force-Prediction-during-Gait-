#!/usr/bin/env python
# coding: utf-8

#%%
import pandas as pd
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from numpy import array
from keras.layers import Input,LSTM, GRU
from keras.models import Model
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.stats import wilcoxon
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler

#%%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
strategy = tf.distribute.MirroredStrategy()

#%% read files and add column titles

import os
# Get a list of all files in the folder
input_test_dier = 'Input/Test/' 
input_test = os.listdir(input_test_dier)
output_test_dier = 'Output/Test/' 
output_test = os.listdir(output_test_dier)
input_train_dier = 'Input/Train/' 
input_train = os.listdir(input_train_dier)
output_train_dier = 'Output/Train/' 
output_train = os.listdir(output_train_dier)

input_files_path_test = pd.DataFrame()
output_files_path_test = pd.DataFrame()
for dier in input_test:
    dier1 = input_test_dier + dier + '/'
    input_files_path_test = pd.concat([input_files_path_test, pd.DataFrame([dier1])])
    
for dier in output_test:
    dier1 = output_test_dier + dier + '/'
    output_files_path_test = pd.concat([output_files_path_test, pd.DataFrame([dier1])])

input_files_path_train = pd.DataFrame()
output_files_path_train = pd.DataFrame()
for dier in input_train:
    dier1 = input_train_dier + dier + '/'
    input_files_path_train = pd.concat([input_files_path_train, pd.DataFrame([dier1])])
    
for dier in output_train:
    dier1 = output_train_dier + dier + '/'
    output_files_path_train = pd.concat([output_files_path_train, pd.DataFrame([dier1])])

input_files_path_test = input_files_path_test.reset_index().sort_values(by = 0, ascending=True)
output_files_path_test = output_files_path_test.reset_index().sort_values(by = 0, ascending=True) 
input_files_path_train = input_files_path_train.reset_index().sort_values(by = 0, ascending=True)
output_files_path_train = output_files_path_train.reset_index().sort_values(by = 0, ascending=True) 
#%%
for z in range(len(input_files_path_train)):
    # z = 0  
    files_list_input_train = os.listdir(input_files_path_train[0].iloc[z])
    files_list_input_train = np.array(files_list_input_train)
    dier_input_train = input_files_path_train[0].iloc[z]
    
    files_list_input_test = os.listdir(input_files_path_test[0].iloc[z])
    files_list_input_test = np.array(files_list_input_test)
    dier_input_test = input_files_path_test[0].iloc[z]

    files_list_output_train = os.listdir(output_files_path_train[0].iloc[z])
    files_list_output_train = np.array(files_list_output_train)
    dier_output_train = output_files_path_train[0].iloc[z]
    
    files_list_output_test = os.listdir(output_files_path_test[0].iloc[z])
    files_list_output_test = np.array(files_list_output_test)
    dier_output_test = output_files_path_test[0].iloc[z]

    
    #
    for i in range(len(files_list_input_train)):
        # l = 0
        # i=0
        # i = 0
        print(str(files_list_input_train[i]))
        print(str(files_list_output_train[i]))
        
        # scaler = MinMaxScaler(feature_range=(0, 1))
        ## import data from sheet 
        input_train = pd.read_excel(dier_input_train + str(files_list_input_train[i]), header = None, skiprows=1)
        input_test = pd.read_excel(dier_input_test + str(files_list_input_test[i]), header = None, skiprows=1)
        output_train = pd.read_excel(dier_output_train + str(files_list_output_train[i]), header = None, skiprows=1)
        output_test = pd.read_excel(dier_output_test + str(files_list_output_test[i]), header = None , skiprows=1)    
        
        ## drop columns with more than 100 nan entrys
        threshold = 100
        drop_cols = [col for col in output_train.columns if output_train[col].isnull().sum() >= threshold]
        output_train.drop(drop_cols, axis=1, inplace=True)
        input_train.drop(drop_cols, axis=1, inplace=True)
        
        threshold = 100
        drop_cols_test = [col for col in output_test.columns if output_test[col].isnull().sum() >= threshold]
        output_test.drop(drop_cols_test, axis=1, inplace=True)
        input_test.drop(drop_cols_test, axis=1, inplace=True)
        
        length = len(input_train)
        # Fill NaN values with the last non-null value in the same column
        input_train = input_train.reindex(range(length)).fillna(0)
        input_test = input_test.reindex(range(length)).fillna(0)
        output_train = output_train.reindex(range(length)).fillna(0)
        output_test = output_test.reindex(range(length)).fillna(0)
              
        
        # # Fill NaN values with the last non-null value in the same column
        # input_train = input_train.fillna(method='ffill', axis=1)
        # input_test = input_test.fillna(method='ffill', axis=1)
        # output_train = output_train.fillna(method='ffill', axis=1)
        # output_test = output_test.fillna(method='ffill', axis=1)
             
       # # Find the shortest length DataFrame
       #  shortest_df = min(input_test, input_train,output_test,output_train , key=len)
         
       #  # length of shortest df
       #  input_train = input_train.iloc[:len(shortest_df)]
       #  input_test = input_test.iloc[:len(shortest_df)]
       #  output_train = output_train.iloc[:len(shortest_df)]
       #  output_test = output_test.iloc[:len(shortest_df)]
        a  = length 
             
        input_train = input_train.T
        input_test = input_test.T
        output_train = output_train.T
        output_test = output_test.T
        ## train the model
        samples_in = input_train.shape[0]
        timesteps_in = input_train.shape[1]
        features_in = 1
        sequence_in = input_train.values.reshape((samples_in, timesteps_in, features_in))
        
        samples_out = output_train.shape[0]
        timesteps_out = output_train.shape[1]
        features_out = 1
        sequence_out = output_train.values.reshape((samples_out, timesteps_out, features_out))
        print(sequence_in.shape)
        print(sequence_out.shape)    
        ## model
        # import keras_tuner
        # from keras.callbacks import EarlyStopping,ModelCheckpoint
        # from tensorflow.keras import layers
        # from kerastuner.tuners import BayesianOptimization
        # import keras
        import keras_tuner
        from keras.callbacks import EarlyStopping,ModelCheckpoint
        from tensorflow.keras import layers
        from keras_tuner.tuners import BayesianOptimization
        import keras
        from keras.models import Sequential
        # from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
        from tensorflow.keras.optimizers import Adam, SGD, RMSprop
        from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, Huber
        # Define the model builder function
        def build_model(hp):
            with strategy.scope():
                 model = Sequential()
                 ## to use GPU avoid Relu its Leaky
                 # model.add(GRU(units=32, activation=hp.Choice('activation', ['relu', 'tanh']), input_shape=(timesteps_in,1),return_sequences=True))
                 model.add(GRU(units=32, activation=hp.Choice('activation', ['tanh']), input_shape=(timesteps_in,1),return_sequences=True))
                 for l in range(hp.Int('num_layers', 1, 10)):
                     model.add(GRU(units=hp.Int('units_' + str(l), min_value=32, max_value=512, step=32), activation=hp.Choice('activation', ['relu', 'tanh']), return_sequences=True))
                 model.add(Dense(units=1)) 
                 optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
                 # optimizer = hp.Choice('optimizer', ['adam'])
                 learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
                 if optimizer == 'adam':
                    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                 elif optimizer == 'sgd':
                    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
                 else:
                    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
                 loss = hp.Choice('loss', ['mse', 'mae', 'huber_loss'])
                 model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
                 return model  
        # Define the tuner
        tuner = BayesianOptimization(
            build_model,
            objective='val_mae',
            max_trials=10,
            directory='Optimized_model',
            project_name= dier_output_train[-22:-6] + str(files_list_output_train[i])[:-5],
            overwrite=True
        )
        callbacks = [EarlyStopping(monitor='val_mae',mode="min",patience=5, restore_best_weights=True)]
        # Search for the best hyperparameters
        tuner.search(sequence_in, sequence_out, epochs=1000, validation_split=0.1, shuffle=True,callbacks=callbacks,verbose=1)
        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        # Get the hyperparameters of the best model
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        # Convert the hyperparameters to a DataFrame
        hyperparameters_df = pd.DataFrame([best_hyperparameters.values])
        
        
        # Save the hyperparameters to a CSV file
        hyperparameters_df.to_excel('Hyperparameters/' + dier_output_train[-22:-6] + str(files_list_input_train[i])[:-5] + 'best_hyperparameters.xlsx', index_label='Hyperparameter')
        model =  tuner.hypermodel.build(best_hyperparameters)
        # model = build_model(keras_tuner.HyperParameters())
        model.summary()
        # Save the model summary to a text file
        with open('Summary/'+dier_output_train[-22:-6] + str(files_list_input_train[i])[:-5] +'Best_model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        ## fit the model
        model.fit(sequence_in, sequence_out, epochs=1000, validation_split=0.1, shuffle=True,callbacks=callbacks,verbose=1)
        ## save model
        model.save('Model/'+dier_output_train[-22:-6] + str(files_list_input_train[i])[:-5] + '_epoch_' + '.h5')
        # model testing
        samples_test = input_test.shape[0]
        timesteps_test = input_test.shape[1]
        features_test = 1
        sequence_test = input_test.values.reshape((samples_test, timesteps_test, features_test))
        samples_base = output_test.shape[0]
        timesteps_base = output_test.shape[1]
        features_base = 1
        sequence_base = output_test.values.reshape((samples_base, timesteps_base, features_base))
        pred = model.predict(sequence_test, verbose=0)
        rmse_1 = pd.DataFrame()
        nrmse_1 = pd.DataFrame()
        for k in range(samples_test):
            # k= 0
            ind = k
            x = np.linspace(0,a-1,a)
            x_in = sequence_test[ind,:,0]
            y_pred = pred[ind,:,0]
            y_true = sequence_base[ind,:,0]
            # plt.plot(x,x_in,label ="Ankle Position")
            plt.plot(x,y_pred,label ="Predicted Hip JRF (pred)")
            plt.plot(x,y_true,label ="Measured Hip JRF (expt)")
            plt.legend()
            plt.savefig('Figure/'+ dier_output_train[-22:-6] + str(files_list_input_train[i])[:-5] + '_'+ str(k)+ '_epoch_' + '.png')
            # plt.figure()
            plt.close()
            # # Calculate the RMSE
            # mse_base = np.square(np.subtract(y_true, y_pred)).mean()
            # rmse_base = sqrt(mse_base) 
            # # Calculate the nRMSE
            # nrmse_base = rmse_base / (y_true.max() - y_true.min())
            # Calculate the RMSE
            mse_pred = np.square(np.subtract(y_true, y_pred)).mean()
            rmse_pred = sqrt(mse_pred)
            # Calculate the nRMSE
            nrmse_pred = rmse_pred / (y_true.max() - y_true.min())
            # save as dataFrame
            header2 = pd.DataFrame(["ForcePlate*Seq2Seq_LSTM"])
            rmse = pd.DataFrame([rmse_pred]).transpose()
            rmse.set_axis(["ForcePlate*Seq2Seq_LSTM"], axis=1)
            rmse_1 = rmse
            # pd.concat([rmse_1,rmse], axis=1)
            nrmse = pd.DataFrame([nrmse_pred]).transpose()
            nrmse.set_axis(["ForcePlate*Seq2Seq_LSTM"], axis=1)       
            nrmse_1 = nrmse
            # pd.concat([nrmse_1,nrmse], axis=1)
            ## Save predictions to excel
            headers = pd.DataFrame(["ForcePlate", 'Seq2Seq_LSTM'])
            out = pd.DataFrame([y_true, y_pred]).transpose()
            out.set_axis(["ForcePlate", 'Seq2Seq_LSTM'], axis=1)
            # writer = pd.ExcelWriter(str(input_files[i])[6:-4] +"_epoch_"+ str(epoche)+ '.xlsx', engine='xlsxwriter')
            writer = pd.ExcelWriter('Stat/'+dier_output_train[-22:-6] + str(files_list_input_train[i])[:-5] +"_model_output_"+ str(k) +'.xlsx', engine='xlsxwriter')
            out.to_excel(writer, sheet_name='Raw_data', header = True) 
            rmse_1.to_excel(writer, sheet_name='RMSE', header = True)
            nrmse_1.to_excel(writer, sheet_name='NRMSE', header = True)
            # Wilcoxon_pvalue.to_excel(writer, sheet_name='Wilcoxon pValue')
            # model_accuracy.to_excel(str(input_files[i]) + '.xlsx', sheet_name='Model Accuracy')
            headers.to_excel(writer, sheet_name='Header')
            header2.to_excel(writer, sheet_name='Header2')
            # Save the Excel file
            writer.close()
            K.clear_session()          