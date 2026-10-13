# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:43:01 2023

@author: hulle
"""



import os
import pandas as pd


#folder_path = 'AP LEFT/ANKLE-HEEL/'
#folder_path = 'AP LEFT/ANKLE-KNEE/'
#folder_path = 'AP LEFT/HEEL/'
#folder_path = 'AP LEFT/KNEE/'
#folder_path = 'AP LEFT/ANKLE-HIP/'


#folder_path = 'AP RIGHT/ANKLE-HEEL/'
# folder_path = 'AP RIGHT/ANKLE-KNEE/'
# folder_path = 'AP RIGHT/HEEL/'
#folder_path = 'AP RIGHT/KNEE/'
#folder_path = 'AP RIGHT/ANKLE-HIP/'

# folder_path = 'ML LEFT/ANKLE-HEEL/'
#folder_path = 'ML LEFT/ANKLE-KNEE/'
#folder_path = 'ML LEFT/HEEL/'
#folder_path = 'ML LEFT/KNEE/'
#folder_path = 'ML LEFT/ANKLE-HIP/'

#folder_path = 'ML RIGHT/ANKLE-HEEL/'
#folder_path = 'ML RIGHT/ANKLE-KNEE/'
#folder_path = 'ML RIGHT/HEEL/'
#folder_path = 'ML RIGHT/KNEE/'
#folder_path = 'ML RIGHT/ANKLE-HIP/'

#folder_path = 'LONG LEFT/ANKLE-HEEL/'
#folder_path = 'LONG LEFT/ANKLE-KNEE/'
#folder_path = 'LONG LEFT/HEEL/'
#folder_path = 'LONG LEFT/KNEE/'
#folder_path = 'LONG LEFT/ANKLE-HIP/'

#folder_path = 'LONG RIGHT/ANKLE-HEEL/'
#folder_path = 'LONG RIGHT/ANKLE-KNEE/'
#folder_path = 'LONG RIGHT/HEEL/'
folder_path = 'LONG RIGHT/KNEE/'
#folder_path = 'LONG RIGHT/ANKLE-HIP/'


max_filename_length = 50

# Get a list of all files in the folder
file_list = os.listdir(folder_path)
# reset
file_list_pd = pd.DataFrame()
RMSE_p = pd.DataFrame()
nRMSE_p = pd.DataFrame()
ForcePlate_p = pd.DataFrame()
Seq2Seq_p = pd.DataFrame()
file_name_pd = pd.DataFrame()

# Loop through each file in the folder
for file_name in file_list:
    # Check if the file is an Excel file and the filename is not too long
    if file_name.endswith('.xlsx') and len(file_name) <= max_filename_length: 
        print(file_name)
        df_RMSE = pd.read_excel(os.path.join(folder_path, file_name), sheet_name='RMSE', header=0)
        df_nRMSE = pd.read_excel(os.path.join(folder_path, file_name), sheet_name='NRMSE', header=0)
        df_raw = pd.read_excel(os.path.join(folder_path, file_name), sheet_name='Raw_data', header=0)
        # df_nRMSE = pd.read_excel(os.path.join(folder_path, 'L_ANKLE_IN_fast_0_model_output_1.xlsx'), sheet_name='NRMSE', header=0)
        RMSE_base =  df_RMSE.loc[:,0]
        # RMSE_pred = df_RMSE.loc[:,1]
        nRMSE_base =  df_nRMSE.loc[:,0]
        # nRMSE_pred = df_nRMSE.loc[:,1]
        ForcePlate = df_raw.loc[:,0]
        Seq2Seq = df_raw.loc[:,1]
        # Pred = df_raw.loc[:,2]
        RMSE_p = pd.concat([RMSE_p, pd.DataFrame([RMSE_base])], axis = 1)
        # RMSE_p = pd.concat([RMSE_p, pd.DataFrame([RMSE_pred])], axis = 1)
        nRMSE_p = pd.concat([nRMSE_p, pd.DataFrame([nRMSE_base])], axis = 1)
        # nRMSE_p = pd.concat([nRMSE_p, pd.DataFrame([nRMSE_pred])], axis = 1)
        ForcePlate_p = pd.concat([ForcePlate_p, pd.DataFrame([ForcePlate])], axis  =0)
        Seq2Seq_p = pd.concat([Seq2Seq_p, pd.DataFrame([Seq2Seq])], axis  =0)
        # Pred_p = pd.concat([Pred_p, pd.DataFrame([Pred])], axis  =0)
        file_name_pd = pd.concat([file_name_pd, pd.DataFrame([str(file_name)[0:-5]])], axis =1)
        
ForcePlate_p = ForcePlate_p.transpose()
Seq2Seq_p = Seq2Seq_p.transpose()
# Pred_p = Pred_p.transpose()       
writer = pd.ExcelWriter("Analysis" + folder_path.replace('/','_') +'.xlsx', engine='xlsxwriter')
RMSE_p.to_excel(writer, sheet_name='RMSE', header = True) 
# RMSE_p.to_excel(writer, sheet_name='RMSE_pred', header = True) 
nRMSE_p.to_excel(writer, sheet_name='nRMSE', header = True) 
# nRMSE_p.to_excel(writer, sheet_name='nRMSE_pred', header = True) 
ForcePlate_p.to_excel(writer, sheet_name='Measured', header = True) 
Seq2Seq_p.to_excel(writer, sheet_name='Predicted', header = True) 
file_name_pd.to_excel(writer, sheet_name='Name', header = True)
# Save the Excel file
writer.close()
        
                
        
        

        

    


