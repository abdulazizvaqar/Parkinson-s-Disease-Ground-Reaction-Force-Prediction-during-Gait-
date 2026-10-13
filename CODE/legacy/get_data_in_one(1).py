# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:16:10 2024

@author: hulle
"""

# excluded = [1, 2, 3, 4, 10, 12, 14, 18, 19, 21, 24, 25, 26, 28, 32, 40, 45, 48, 51, 52, 56, 58, 61, 74, 79, 87, 94, 95, 100, 102, 104, 112, 113 ]

#%%

#%%
import pandas as pd
import os

# Define the folder containing the Excel files
# root = ['D:/Predict Kinetics/External Validation GRF/Predictions/Seq2Seq/Long Left/1D-Convo/',
#         'D:/Predict Kinetics/External Validation GRF/Predictions/Seq2Seq/Long Left/GRU/',
#         'D:/Predict Kinetics/External Validation GRF/Predictions/Seq2Seq/Long Left/LSTM/']

# root = ['D:/Predict Kinetics/External Validation GRF/Predictions/Seq2Seq/Long Right/1D-Convo/']
# #         # ,
# root =        ['D:/Predict Kinetics/External Validation GRF/Predictions/Seq2Seq/Long Right/GRU/',
#         'D:/Predict Kinetics/External Validation GRF/Predictions/Seq2Seq/Long Right/LSTM/']
        
# root = ['D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/1D-Convo/',
#         'D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/GRU/',
#         'D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Left/LSTM/']

# root = ['D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/1D-Convo/']
        # ,
root =    [ 'D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/GRU/']
           # ,
#         'D:/Predict Kinetics/External Validation GRF/Predictions/PINN/Long Right/LSTM/']


# Define the range of indices and the excluded indices
indices = range(122)
# excluded = [1, 2, 3, 4,5, 8,10,11, 12, 13, 15, 19,20, 22, 25, 26, 27, 29, 40, 41,42,  46, 49, 51, 52, 53, 54,55, 57, 58, 59, 60, 61,62, 64, 67,69,71, 75, 80,81, 88, 89, 95,96,97, 101, 103,105, 106, 107, 108,109, 111, 112, 113, 114,116,121 ]

excluded = [3, 4,6, 9, 10,12, 13,15, 19,25,29,30,33,35,37,38,39,40,49, 53,57,62,63,64,65,68,69,70,72,74,76,78, 79,83,
            85,87,88,89,91,92,93,94,95,101,102,103,109,111,112,113,115,117,121]
# Initialize empty DataFrames to store the concatenated data
predicted_data = pd.DataFrame()
measured_data = pd.DataFrame()

#%%
# Iterate over each folder in the root directory
for root_directory in root:
    
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)
        
        # Check if it is a directory
        if os.path.isdir(folder_path):
            # Initialize empty DataFrames to store the concatenated data
            predicted_data = pd.DataFrame()
            measured_data = pd.DataFrame()
    
            # Loop through the indices, skipping the excluded ones
            for i in indices:
                if i not in excluded:
                # file_name = f'Long LeftLNormalisedGRFseq2seq{i}.xlsx'
                    # file_name = f'Long LeftLNormalisedGRFPINN{i}.xlsx'
                    # file_name = f'Long RightRNormalisedGRFseq2seq{i}.xlsx'
                    # file_name = f'seq2seq{i}.xlsx'
                    # file_name = f'pinn{i}.xlsx'
                    file_name =f'Long RightRNormalisedGRFPINN{i}.xlsx'
                    file_path = os.path.join(folder_path, file_name)
                    
                    if os.path.exists(file_path):
                        # Read the Excel file
                        df = pd.read_excel(file_path)
                        
                        # Extract the second and third columns
                        predicted_df = df.iloc[:, [1]]
                        measured_df = df.iloc[:, [2]]
                        
                        # Rename columns to keep consistent column names
                        predicted_df.columns = [f'Predicted_{i}']
                        measured_df.columns = [f'Measured_{i}']
                        
                        # Concatenate the data
                        predicted_data = pd.concat([predicted_data, predicted_df], axis=1, ignore_index=False)
                        measured_data = pd.concat([measured_data, measured_df], axis=1, ignore_index=False)
            
            # Display or save the concatenated DataFrames
            print(f"Processed folder: {folder_name}")
            print(predicted_data)
            print(measured_data)
    
            # Save to separate Excel sheets
            output_file = os.path.join(root_directory+'/'+folder_path[-47:].replace('/','_')+'_'+'concatenated_data.xlsx')
            with pd.ExcelWriter(output_file) as writer:
                predicted_data.to_excel(writer, sheet_name='Predicted', index=False)
                measured_data.to_excel(writer, sheet_name='Measured', index=False)




