# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 20:35:43 2023

@author: lme19
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def clean_data_high(path, reference):
    
    df_original = pd.read_csv(path, skiprows = 8, delimiter = '\t')

    date_and_time = df_original['# Date (DD/MM/YYYY)'] + ' ' + df_original['Time (hh:mm:ss)']

    datetime = pd.to_datetime(df_original['# Date (DD/MM/YYYY)'] + ' ' + df_original['Time (hh:mm:ss)'])

    df_original['Date_Time'] = datetime
    
    df_no_reference = df_original[df_original.Date_Time != reference]
    
    df_high_scans = df_no_reference[df_no_reference.Scans > 30]

    lower_lim = 2930
    upper_lim = 3361
    
    #This goes between SZA 85.009651 and SZA 94.965416
    #for the best parameters DOAS run
    #refine later to pick out the indexes closest to 85 and 95 for each dataset
    #if possible
    
    #ATTENTION
    #STUFF TO IMPROVE HERE
    #READ ABOVE
    

    #separate sections for waves and SCD

    stable_Datetime = df_high_scans['Date_Time'][lower_lim:upper_lim]
    
    stable_SZA = df_high_scans['SZA'][lower_lim:upper_lim]
    
    stable_SlCol_O3 = df_high_scans['Test.SlCol(O3)'][lower_lim:upper_lim]
    stable_SlErr_O3 = 2 * df_high_scans['Test.SlErr(O3)'][lower_lim:upper_lim]
    
    stable_SlCol_NO2 = df_high_scans['Test.SlCol(NO2)'][lower_lim:upper_lim]
    stable_SlErr_NO2 = 2 * df_high_scans['Test.SlErr(NO2)'][lower_lim:upper_lim]
    
    stable_RMS = df_high_scans['Test.RMS'][lower_lim:upper_lim]
    
    stable_CLD = df_high_scans['Test.SlCol(CLD)'][lower_lim:upper_lim]
    stable_CLD_error = 1 * df_high_scans['Test.SlErr(CLD)'][lower_lim:upper_lim]   
    
    stable_SlCol_IO = df_high_scans['Test.SlCol(IO)'][lower_lim:upper_lim]
    stable_SlErr_IO = 2 * df_high_scans['Test.SlErr(IO)'][lower_lim:upper_lim]
    
    stable_SlCol_O4 = df_high_scans['Test.SlCol(O4)'][lower_lim:upper_lim]
    stable_SlErr_O4 = 2 * df_high_scans['Test.SlErr(O4)'][lower_lim:upper_lim]
    

    new_array = np.vstack((stable_SZA, stable_SlCol_O3, stable_SlErr_O3, stable_SlCol_NO2, stable_SlErr_NO2, stable_RMS, stable_CLD, stable_CLD_error, stable_SlCol_IO, stable_SlErr_IO, stable_SlCol_O4, stable_SlErr_O4))        
    
#    plt.rcParams.update({'font.size': 5})
      
    # Initialise the subplot function using number of rows and columns
#    figure, axis = plt.subplots(5, 1)
      
#    axis[0].errorbar(stable_SZA, stable_SlCol_O3,  yerr = stable_SlErr_O3, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', color = 'black', capsize = 1, markersize = 0.5)
#    axis[0].set_title('SlCol(O3)')
    #axis[0].yscale
#    axis[0].grid()
      

#    axis[1].errorbar(stable_SZA, stable_SlCol_NO2, yerr = stable_SlErr_NO2, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', capsize = 1, markersize = 0.5, color = 'black')
#    axis[1].set_title('SlCol(NO2)')
#    axis[1].grid()

#    axis[2].errorbar(stable_SZA, stable_SlCol_IO, yerr = stable_SlErr_IO, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', capsize = 1, markersize = 0.5, color = 'black')
#    axis[2].set_title('SlCol(IO)')
#    axis[2].grid()
      

#    axis[3].plot(stable_SZA, stable_RMS)
#    axis[3].set_title('RMS')
#    axis[3].grid()
      

#    axis[4].errorbar(stable_SZA, stable_CLD, yerr = stable_CLD_error, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', capsize = 1, markersize = 0.5, color = 'black')
#    axis[4].set_title("CLD")
#    axis[4].grid()

#    plt.subplots_adjust(hspace=0.7)
#    plt.xlabel('SZA (degrees)')

      
    # Combine all the operations and display
#    plt.savefig(path[0:20], dpi=300)

#    plt.show()

    
    return new_array
    
   
def clean_data_low(path, reference):
    
    df_original = pd.read_csv(path, skiprows = 8, delimiter = '\t')

    date_and_time = df_original['# Date (DD/MM/YYYY)'] + ' ' + df_original['Time (hh:mm:ss)']

    datetime = pd.to_datetime(df_original['# Date (DD/MM/YYYY)'] + ' ' + df_original['Time (hh:mm:ss)'])

    df_original['Date_Time'] = datetime
    
    df_no_reference = df_original[df_original.Date_Time != reference]
    
    df_high_scans = df_no_reference[df_no_reference.Scans > 30]

    lower_lim = 1103
    upper_lim = 2929
    
    #These limits go from 55 - 85 SZA roughly

    #separate sections for waves and SCD

    stable_Datetime = df_high_scans['Date_Time'][lower_lim:upper_lim]
    
    stable_SZA = df_high_scans['SZA'][lower_lim:upper_lim]
    
    stable_SlCol_O3 = df_high_scans['Test.SlCol(O3)'][lower_lim:upper_lim]
    stable_SlErr_O3 = 2 * df_high_scans['Test.SlErr(O3)'][lower_lim:upper_lim]
    
    stable_SlCol_NO2 = df_high_scans['Test.SlCol(NO2)'][lower_lim:upper_lim]
    stable_SlErr_NO2 = 2 * df_high_scans['Test.SlErr(NO2)'][lower_lim:upper_lim]
    
    stable_RMS = df_high_scans['Test.RMS'][lower_lim:upper_lim]
    
    stable_CLD = df_high_scans['Test.SlCol(CLD)'][lower_lim:upper_lim]
    stable_CLD_error = 1 * df_high_scans['Test.SlErr(CLD)'][lower_lim:upper_lim]   
    
    # changed the CLD error to only be once
    
    stable_SlCol_IO = df_high_scans['Test.SlCol(IO)'][lower_lim:upper_lim]
    stable_SlErr_IO = 2 * df_high_scans['Test.SlErr(IO)'][lower_lim:upper_lim]
    
    stable_SlCol_O4 = df_high_scans['Test.SlCol(O4)'][lower_lim:upper_lim]
    stable_SlErr_O4 = 2 * df_high_scans['Test.SlErr(O4)'][lower_lim:upper_lim]
    

    new_array = np.vstack((stable_SZA, stable_SlCol_O3, stable_SlErr_O3, stable_SlCol_NO2, stable_SlErr_NO2, stable_RMS, stable_CLD, stable_CLD_error, stable_SlCol_IO, stable_SlErr_IO, stable_SlCol_O4, stable_SlErr_O4))        
    
    #plt.rcParams.update({'font.size': 5})
      
    # Initialise the subplot function using number of rows and columns
    #figure, axis = plt.subplots(5, 1)
      
    #axis[0].errorbar(stable_SZA, stable_SlCol_O3,  yerr = stable_SlErr_O3, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', color = 'black', capsize = 1, markersize = 0.5)
    #axis[0].set_title('SlCol(O3)')
    #axis[0].yscale
    #axis[0].grid()
      

    #axis[1].errorbar(stable_SZA, stable_SlCol_NO2, yerr = stable_SlErr_NO2, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', capsize = 1, markersize = 0.5, color = 'black')
    #axis[1].set_title('SlCol(NO2)')
    #axis[1].grid()

    #axis[2].errorbar(stable_SZA, stable_SlCol_IO, yerr = stable_SlErr_IO, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', capsize = 1, markersize = 0.5, color = 'black')
    #axis[2].set_title('SlCol(IO)')
    #axis[2].grid()
      

    #axis[3].plot(stable_SZA, stable_RMS)
    #axis[3].set_title('RMS')
    #axis[3].grid()
      

    #axis[4].errorbar(stable_SZA, stable_CLD, yerr = stable_CLD_error, linewidth = 0, ecolor = 'red', elinewidth = 0.1, marker = '.', capsize = 1, markersize = 0.5, color = 'black')
    #axis[4].set_title("CLD")
    #axis[4].grid()

    #plt.subplots_adjust(hspace=0.7)
    #plt.xlabel('SZA (degrees)')

      
    # Combine all the operations and display
    #plt.savefig(path[0:20], dpi=300)

    #plt.show()

    
    return new_array
        
   


best_params_with_IO_run2_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\best_params_with_IO_run2.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_428_468_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_428_468.ASC', '2022-08-23 22:00:05')
polynomial_order_8_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_8_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_7_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_7_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_6_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_6_redo.ASC', '2022-08-23 22:00:05')
#polynomial_order_5_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_5_redo.ASC', '2022-08-23 22:00:05')
#why don't I have a 5
#lol
polynomial_order_4_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_4_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_with_CLD_sigmas_v2_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_with_CLD_sigmas_v2.ASC', '2022-08-23 22:00:05')
polynom_order_2_with_CLD_sigmas_v2_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_2_with_CLD_sigmas_v2.ASC', '2022-08-23 22:00:05')
ref_224504_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\ref_224504_redo.ASC', '2022-08-23 22:45:04')
ref_221505_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\ref_221505_redo.ASC', '2022-08-23 22:15:05')
ref_223000_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\ref_223000_redo.ASC', '2022-08-23 22:30:00')
ref_215959_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\ref_215959_redo.ASC', '2022-08-23 21:59:59')
ref_230002_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\ref_230002_redo.ASC', '2022-08-23 23:00:02')
ref_210005_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\ref_210005_redo.ASC', '2022-08-23 21:00:05')
linear_offset_order_2_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\linear_offset_order_2_redo.ASC', '2022-08-23 22:00:05')
NO2_294K_Vandaele_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_294K_Vandaele_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_293K_Burrows_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_293K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_293K_Bogumil_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_293K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_273K_Burrows_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_273K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_273K_Bogumil_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_273K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_243K_Bogumil_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_243K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_241K_Burrows_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_241K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_223K_Bogumil_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_223K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_221K_Burrows_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_221K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_203K_Bogumi_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_203K_Bogumi_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_221K_Burrows_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_221K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_220K_Vandaele_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\NO2_220K_Vandaele_redo.ASC', '2022-08-23 22:00:05')
O3_193K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_193K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_203K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_203K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_213K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_213K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_223K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_223K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_233K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_233K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_273K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_273K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_253K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_253K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_263K_ref_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\O3_263K_ref_redo.ASC', '2022-08-23 22:00:05')
no_shift_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\no_shift_redo.ASC', '2022-08-23 22:00:05')
no_stretch_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\no_stretch_redo.ASC', '2022-08-23 22:00:05')
first_order_stretch_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\first_order_stretch_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_435_465_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_435_465_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_425_455_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_425_455_redo.ASC', '2022-08-23 22:00:05')
best_params_sensitivity_polynom_order3_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\best_params_sensitivity_polynom_order3.ASC', '2022-08-23 22:00:05')
best_params_sensitivity_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\best_params_sensitivity.ASC', '2022-08-23 22:00:05')
polynom_order_5_with_CLD_sigmas_v2_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_5_with_CLD_sigmas_v2.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_410_450_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_410_450_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_420_460_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_420_460_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_430_470_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_430_470_redo.ASC', '2022-08-23 22:00:05')
PCA_poly2_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\PCA_poly2.ASC', '2022-08-23 22:00:05')
PCA_poly3_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\PCA_poly3.ASC', '2022-08-23 22:00:05')
no_PCA_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\no_PCA_v2.ASC', '2022-08-23 22:00:05')
RingSpectrum_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\RingSpectrum.ASC', '2022-08-23 22:00:05')



#%%

best_params_with_IO_run2_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\best_params_with_IO_run2.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_428_468_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_428_468.ASC', '2022-08-23 22:00:05')
polynomial_order_8_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynomial_order_8_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_7_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynomial_order_7_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_6_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynomial_order_6_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_4_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynomial_order_4_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_with_CLD_sigmas_v2_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_with_CLD_sigmas_v2.ASC', '2022-08-23 22:00:05')
polynom_order_2_with_CLD_sigmas_v2_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_2_with_CLD_sigmas_v2.ASC', '2022-08-23 22:00:05')
ref_224504_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\ref_224504_redo.ASC', '2022-08-23 22:45:04')
ref_221505_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\ref_221505_redo.ASC', '2022-08-23 22:15:05')
ref_223000_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\ref_223000_redo.ASC', '2022-08-23 22:30:00')
ref_215959_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\ref_215959_redo.ASC', '2022-08-23 21:59:59')
ref_230002_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\ref_230002_redo.ASC', '2022-08-23 23:00:02')
ref_210005_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\ref_210005_redo.ASC', '2022-08-23 21:00:05')
linear_offset_order_2_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\linear_offset_order_2_redo.ASC', '2022-08-23 22:00:05')
NO2_294K_Vandaele_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_294K_Vandaele_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_293K_Burrows_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_293K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_293K_Bogumil_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_293K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_273K_Burrows_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_273K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_273K_Bogumil_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_273K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_243K_Bogumil_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_243K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_241K_Burrows_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_241K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_223K_Bogumil_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_223K_Bogumil_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_221K_Burrows_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_221K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_203K_Bogumi_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_203K_Bogumi_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_221K_Burrows_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_221K_Burrows_ref_redo.ASC', '2022-08-23 22:00:05')
NO2_220K_Vandaele_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\NO2_220K_Vandaele_redo.ASC', '2022-08-23 22:00:05')
O3_193K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_193K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_203K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_203K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_213K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_213K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_223K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_223K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_233K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_233K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_273K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_273K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_253K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_253K_ref_redo.ASC', '2022-08-23 22:00:05')
O3_263K_ref_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\O3_263K_ref_redo.ASC', '2022-08-23 22:00:05')
no_shift_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\no_shift_redo.ASC', '2022-08-23 22:00:05')
no_stretch_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\no_stretch_redo.ASC', '2022-08-23 22:00:05')
first_order_stretch_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\first_order_stretch_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_435_465_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_435_465_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_425_455_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_425_455_redo.ASC', '2022-08-23 22:00:05')
best_params_sensitivity_polynom_order3_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\best_params_sensitivity_polynom_order3.ASC', '2022-08-23 22:00:05')
best_params_sensitivity_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\best_params_sensitivity.ASC', '2022-08-23 22:00:05')
polynom_order_5_with_CLD_sigmas_v2_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_5_with_CLD_sigmas_v2.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_410_450_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_410_450_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_420_460_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_420_460_redo.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_430_470_redo_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_430_470_redo.ASC', '2022-08-23 22:00:05')
PCA_poly2_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\PCA_poly2.ASC', '2022-08-23 22:00:05')
PCA_poly3_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\PCA_poly3.ASC', '2022-08-23 22:00:05')
no_PCA_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\no_PCA_v2.ASC', '2022-08-23 22:00:05')
RingSpectrum_low = clean_data_low(r'C:\Users\lme19\Documents\Test_data\RingSpectrum.ASC', '2022-08-23 22:00:05')

#Data indexing key
# 0 = stable_SZA
# 1 = stable_SlCol_O3
# 2 = stable_SlErr_O3
# 3 = stable_SlCol_NO2
# 4 = stable_SlErr_NO2
# 5 = stable_RMS
# 6 = stable_CLD
# 7 = stable_CLD_error
# 8 = stable_SlCol_IO
# 9 = stable_SlErr_IO

#%%


plt.rcParams.update({'font.size': 5, 'errorbar.capsize' : 0, 'lines.markersize' : 0.8, 'lines.marker' : '.', 'lines.linewidth': 0.2})

# All IO col densities



#All the RMS (High SZA)

plt.errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[8], yerr = polynom_order_3_wav_428_468_analysis[9], label = 'polynom_order_3_wav_428_468')
plt.errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[8], polynom_order_3_wav_435_465_analysis[9], label = 'polynom_order_3_wav_435_465')
plt.errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[8], polynom_order_3_wav_425_455_analysis[9], label = 'polynom_order_3_wav_425_455')
plt.errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[8], polynom_order_3_wav_410_450_redo_analysis[9], label = 'polynom_order_3_wav_410_450')
plt.errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[8], polynom_order_3_wav_420_460_redo_analysis[9], label = 'polynom_order_3_wav_420_460')
plt.errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[8],polynom_order_3_wav_430_470_redo_analysis[9], label = 'polynom_order_3_wav_430_470')
plt.errorbar(ref_224504_redo_analysis[0], ref_224504_redo_analysis[8], ref_224504_redo_analysis[9], label = 'ref_224504')
plt.errorbar(ref_221505_redo_analysis[0], ref_221505_redo_analysis[8], ref_221505_redo_analysis[9],label = 'ref_221505')
plt.errorbar(ref_223000_redo_analysis[0], ref_223000_redo_analysis[8], ref_223000_redo_analysis[9],label = 'ref_223000')
plt.errorbar(ref_215959_redo_analysis[0], ref_215959_redo_analysis[8], ref_215959_redo_analysis[9], label = 'ref_215959')
plt.errorbar(ref_230002_redo_analysis[0], ref_230002_redo_analysis[8], ref_230002_redo_analysis[9], label = 'ref_230002')
plt.errorbar(ref_210005_redo_analysis[0], ref_210005_redo_analysis[8], ref_210005_redo_analysis[9], label = 'ref_210005')
plt.errorbar(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 8')
plt.errorbar(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 7')
plt.errorbar(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 6')
plt.errorbar(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[8], polynom_order_5_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 5')
plt.errorbar(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[8], polynomial_order_4_redo_analysis[9], label = 'Polynomial Order 4')
plt.errorbar(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[8], polynom_order_3_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 3')
plt.errorbar(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[8], polynom_order_2_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 2')
plt.errorbar(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[8], NO2_294K_Vandaele_ref_redo_analysis[9], label = 'NO2_294K_Vandaele')
plt.errorbar(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[8], NO2_293K_Burrows_ref_redo_analysis[9], label = 'NO2_293K_Burrows')
plt.errorbar(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[8], NO2_293K_Bogumil_ref_redo_analysis[9], label = 'NO2_293K_Bogumil')
plt.errorbar(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[8], NO2_273K_Burrows_ref_redo_analysis[9], label = 'NO2_273K_Burrows')
plt.errorbar(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[8], NO2_273K_Bogumil_ref_redo_analysis[9], label = 'NO2_273K_Bogumil')
plt.errorbar(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[8], NO2_243K_Bogumil_ref_redo_analysis[9], label = 'NO2_243K_Bogumil')
plt.errorbar(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[8],  NO2_241K_Burrows_ref_redo_analysis[9], label = 'NO2_241K_Burrows')
plt.errorbar(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[8], NO2_223K_Bogumil_ref_redo_analysis[9],  label = 'NO2_223K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8],  NO2_221K_Burrows_ref_redo_analysis[9],label = 'NO2_221K_Burrows')
plt.errorbar(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[8], NO2_203K_Bogumi_ref_redo_analysis[9], label = 'NO2_203K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8], NO2_221K_Burrows_ref_redo_analysis[9], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[8], NO2_220K_Vandaele_redo_analysis[9], label = 'NO2_220K_Vandaele')
plt.errorbar(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[8], O3_193K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[8], O3_203K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[8], O3_213K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[8], O3_223K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[8], O3_233K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[8], O3_273K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[8], O3_253K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[8], O3_263K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(no_shift_redo_analysis[0], no_shift_redo_analysis[8], no_shift_redo_analysis[9])
plt.errorbar(no_stretch_redo_analysis[0], no_stretch_redo_analysis[8], no_stretch_redo_analysis[9])
plt.errorbar(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[8], first_order_stretch_redo_analysis[9])
plt.errorbar(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[8], linear_offset_order_2_redo_analysis[9])
plt.errorbar(PCA_poly2_analysis[0], PCA_poly2_analysis[8],PCA_poly2_analysis[9], label = 'Poly2')
plt.errorbar(PCA_poly3_analysis[0], PCA_poly3_analysis[8],PCA_poly3_analysis[9],  label = 'Poly3')
plt.errorbar(no_PCA_analysis[0], no_PCA_analysis[8], no_PCA_analysis[9], label = 'No PCA')
plt.errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], RingSpectrum_analysis[9], label = 'Best Parameters')
plt.title('IO dSCD for All Fit Scenarious (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('dSCD IO')
plt.legend(bbox_to_anchor=(1.0, 1.1), ncol = 2) 

plt.savefig('IO_col_high_SZA', dpi = 400, bbox_inches='tight')

plt.show() 




#%%

plt.errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[8], yerr = polynom_order_3_wav_428_468_analysis[9], label = 'polynom_order_3_wav_428_468', color = 'red')
plt.errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[8], polynom_order_3_wav_435_465_analysis[9], label = 'polynom_order_3_wav_435_465',color = 'red')
plt.errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[8], polynom_order_3_wav_425_455_analysis[9], label = 'polynom_order_3_wav_425_455',color = 'red')
plt.errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[8], polynom_order_3_wav_410_450_redo_analysis[9], label = 'polynom_order_3_wav_410_450',color = 'red')
plt.errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[8], polynom_order_3_wav_420_460_redo_analysis[9], label = 'polynom_order_3_wav_420_460',color = 'red')
plt.errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[8],polynom_order_3_wav_430_470_redo_analysis[9], label = 'polynom_order_3_wav_430_470',color = 'red')
plt.errorbar(ref_224504_redo_analysis[0], ref_224504_redo_analysis[8], ref_224504_redo_analysis[9], label = 'ref_224504', color = 'green')
plt.errorbar(ref_221505_redo_analysis[0], ref_221505_redo_analysis[8], ref_221505_redo_analysis[9],label = 'ref_221505', color = 'green')
plt.errorbar(ref_223000_redo_analysis[0], ref_223000_redo_analysis[8], ref_223000_redo_analysis[9],label = 'ref_223000', color = 'green')
plt.errorbar(ref_215959_redo_analysis[0], ref_215959_redo_analysis[8], ref_215959_redo_analysis[9], label = 'ref_215959', color = 'green')
plt.errorbar(ref_230002_redo_analysis[0], ref_230002_redo_analysis[8], ref_230002_redo_analysis[9], label = 'ref_230002', color = 'green')
plt.errorbar(ref_210005_redo_analysis[0], ref_210005_redo_analysis[8], ref_210005_redo_analysis[9], label = 'ref_210005', color = 'green')
plt.errorbar(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 8', color = 'blue')
plt.errorbar(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 7', color = 'blue')
plt.errorbar(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 6', color = 'blue')
plt.errorbar(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[8], polynom_order_5_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 5', color = 'blue')
plt.errorbar(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[8], polynomial_order_4_redo_analysis[9], label = 'Polynomial Order 4', color = 'blue')
plt.errorbar(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[8], polynom_order_3_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 3', color = 'blue')
plt.errorbar(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[8], polynom_order_2_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 2', color = 'blue')
plt.errorbar(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[8], NO2_294K_Vandaele_ref_redo_analysis[9], label = 'NO2_294K_Vandaele', color = 'purple')
plt.errorbar(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[8], NO2_293K_Burrows_ref_redo_analysis[9], label = 'NO2_293K_Burrows', color = 'purple')
plt.errorbar(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[8], NO2_293K_Bogumil_ref_redo_analysis[9], label = 'NO2_293K_Bogumil', color = 'purple')
plt.errorbar(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[8], NO2_273K_Burrows_ref_redo_analysis[9], label = 'NO2_273K_Burrows', color = 'purple')
plt.errorbar(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[8], NO2_273K_Bogumil_ref_redo_analysis[9], label = 'NO2_273K_Bogumil', color = 'purple')
plt.errorbar(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[8], NO2_243K_Bogumil_ref_redo_analysis[9], label = 'NO2_243K_Bogumil', color = 'purple')
plt.errorbar(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[8],  NO2_241K_Burrows_ref_redo_analysis[9], label = 'NO2_241K_Burrows', color = 'purple')
plt.errorbar(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[8], NO2_223K_Bogumil_ref_redo_analysis[9],  label = 'NO2_223K_Bogumil', color = 'purple')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8],  NO2_221K_Burrows_ref_redo_analysis[9],label = 'NO2_221K_Burrows', color = 'purple')
plt.errorbar(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[8], NO2_203K_Bogumi_ref_redo_analysis[9], label = 'NO2_203K_Bogumil', color = 'purple')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8], NO2_221K_Burrows_ref_redo_analysis[9], label = 'NO2_221K_Burrows', color = 'purple')
plt.errorbar(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[8], NO2_220K_Vandaele_redo_analysis[9], label = 'NO2_220K_Vandaele', color = 'purple')
plt.errorbar(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[8], O3_193K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[8], O3_203K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[8], O3_213K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[8], O3_223K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[8], O3_233K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[8], O3_273K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[8], O3_253K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[8], O3_263K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
plt.errorbar(no_shift_redo_analysis[0], no_shift_redo_analysis[8], no_shift_redo_analysis[9], color = 'orange')
plt.errorbar(no_stretch_redo_analysis[0], no_stretch_redo_analysis[8], no_stretch_redo_analysis[9], color = 'orange')
plt.errorbar(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[8], first_order_stretch_redo_analysis[9], color = 'orange')
plt.errorbar(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[8], linear_offset_order_2_redo_analysis[9], color = 'orange')
plt.errorbar(PCA_poly2_analysis[0], PCA_poly2_analysis[8],PCA_poly2_analysis[9], label = 'Poly2', color = 'pink')
plt.errorbar(PCA_poly3_analysis[0], PCA_poly3_analysis[8],PCA_poly3_analysis[9],  label = 'Poly3', color = 'pink')
plt.errorbar(no_PCA_analysis[0], no_PCA_analysis[8], no_PCA_analysis[9], label = 'No PCA', color = 'pink')
plt.errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], RingSpectrum_analysis[9], label = 'Best Parameters', color = 'pink')
plt.title('IO dSCD for All Fit Scenarious (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('dSCD IO')
plt.legend(bbox_to_anchor=(1.0, 1.1), ncol = 2) 

plt.savefig('IO_col_high_SZA_color_coded', dpi = 400, bbox_inches='tight')

plt.show() 


#%%

# 7 subplots with each varying parameter on its own axes

fig, axs = plt.subplots(4, 2, sharex='col')

plt.suptitle('IO dSCDs under differenert parameter variation')


axs[0, 0].errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[8], yerr = polynom_order_3_wav_428_468_analysis[9], label = 'polynom_order_3_wav_428_468', color = 'red')
axs[0, 0].errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[8], polynom_order_3_wav_435_465_analysis[9], label = 'polynom_order_3_wav_435_465',color = 'red')
axs[0, 0].errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[8], polynom_order_3_wav_425_455_analysis[9], label = 'polynom_order_3_wav_425_455',color = 'red')
axs[0, 0].errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[8], polynom_order_3_wav_410_450_redo_analysis[9], label = 'polynom_order_3_wav_410_450',color = 'red')
axs[0, 0].errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[8], polynom_order_3_wav_420_460_redo_analysis[9], label = 'polynom_order_3_wav_420_460',color = 'red')
axs[0, 0].errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[8],polynom_order_3_wav_430_470_redo_analysis[9], label = 'polynom_order_3_wav_430_470',color = 'red')
axs[0,0].set_ylabel('IO dSCD')
axs[0,0].set_title('Wavelength Range')

axs[1,0].errorbar(ref_224504_redo_analysis[0], ref_224504_redo_analysis[8], ref_224504_redo_analysis[9], label = 'ref_224504', color = 'green')
axs[1,0].errorbar(ref_221505_redo_analysis[0], ref_221505_redo_analysis[8], ref_221505_redo_analysis[9],label = 'ref_221505', color = 'green')
axs[1,0].errorbar(ref_223000_redo_analysis[0], ref_223000_redo_analysis[8], ref_223000_redo_analysis[9],label = 'ref_223000', color = 'green')
axs[1,0].errorbar(ref_215959_redo_analysis[0], ref_215959_redo_analysis[8], ref_215959_redo_analysis[9], label = 'ref_215959', color = 'green')
axs[1,0].errorbar(ref_230002_redo_analysis[0], ref_230002_redo_analysis[8], ref_230002_redo_analysis[9], label = 'ref_230002', color = 'green')
axs[1,0].errorbar(ref_210005_redo_analysis[0], ref_210005_redo_analysis[8], ref_210005_redo_analysis[9], label = 'ref_210005', color = 'green')
axs[1,0].set_ylabel('IO dSCD')
axs[1,0].set_title('Reference Spectrum')

axs[2,0].errorbar(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 8', color = 'blue')
axs[2,0].errorbar(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 7', color = 'blue')
axs[2,0].errorbar(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[8], polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 6', color = 'blue')
axs[2,0].errorbar(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[8], polynom_order_5_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 5', color = 'blue')
axs[2,0].errorbar(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[8], polynomial_order_4_redo_analysis[9], label = 'Polynomial Order 4', color = 'blue')
axs[2,0].errorbar(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[8], polynom_order_3_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 3', color = 'blue')
axs[2,0].errorbar(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[8], polynom_order_2_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 2', color = 'blue')
axs[2,0].set_ylabel('IO dSCD')
axs[2,0].set_title('Polynomial Order')


axs[1,1].errorbar(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[8], NO2_294K_Vandaele_ref_redo_analysis[9], label = 'NO2_294K_Vandaele', color = 'purple')
axs[1,1].errorbar(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[8], NO2_293K_Burrows_ref_redo_analysis[9], label = 'NO2_293K_Burrows', color = 'purple')
axs[1,1].errorbar(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[8], NO2_293K_Bogumil_ref_redo_analysis[9], label = 'NO2_293K_Bogumil', color = 'purple')
axs[1,1].errorbar(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[8], NO2_273K_Burrows_ref_redo_analysis[9], label = 'NO2_273K_Burrows', color = 'purple')
axs[1,1].errorbar(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[8], NO2_273K_Bogumil_ref_redo_analysis[9], label = 'NO2_273K_Bogumil', color = 'purple')
axs[1,1].errorbar(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[8], NO2_243K_Bogumil_ref_redo_analysis[9], label = 'NO2_243K_Bogumil', color = 'purple')
axs[1,1].errorbar(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[8],  NO2_241K_Burrows_ref_redo_analysis[9], label = 'NO2_241K_Burrows', color = 'purple')
axs[1,1].errorbar(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[8], NO2_223K_Bogumil_ref_redo_analysis[9],  label = 'NO2_223K_Bogumil', color = 'purple')
axs[1,1].errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8],  NO2_221K_Burrows_ref_redo_analysis[9],label = 'NO2_221K_Burrows', color = 'purple')
axs[1,1].errorbar(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[8], NO2_203K_Bogumi_ref_redo_analysis[9], label = 'NO2_203K_Bogumil', color = 'purple')
axs[1,1].errorbar(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[8], NO2_220K_Vandaele_redo_analysis[9], label = 'NO2_220K_Vandaele', color = 'purple')
axs[1,1].set_ylabel('IO dSCD')
axs[1,1].set_title('NO2 Reference Spectrum')

axs[2,1].errorbar(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[8], O3_193K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[8], O3_203K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[8], O3_213K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[8], O3_223K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[8], O3_233K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[8], O3_273K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[8], O3_253K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].errorbar(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[8], O3_263K_ref_redo_analysis[9], label = 'O3 ref 193K', color = 'black')
axs[2,1].set_ylabel('IO dSCD')
axs[3,0].set_title('O3 Reference Spectrum')

axs[3,0].errorbar(no_shift_redo_analysis[0], no_shift_redo_analysis[8], no_shift_redo_analysis[9], color = 'orange')
axs[3,0].errorbar(no_stretch_redo_analysis[0], no_stretch_redo_analysis[8], no_stretch_redo_analysis[9], color = 'orange')
axs[3,0].errorbar(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[8], first_order_stretch_redo_analysis[9], color = 'orange')
axs[3,0].errorbar(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[8], linear_offset_order_2_redo_analysis[9], color = 'orange')
axs[3,0].set_xlabel('SZA')
axs[3,0].set_ylabel('IO dSCD')
axs[3,0].set_title('Shift and Stretch')

axs[3,1].errorbar(PCA_poly2_analysis[0], PCA_poly2_analysis[8],PCA_poly2_analysis[9], label = 'Poly2', color = 'pink')
axs[3,1].errorbar(PCA_poly3_analysis[0], PCA_poly3_analysis[8],PCA_poly3_analysis[9],  label = 'Poly3', color = 'pink')
axs[3,1].errorbar(no_PCA_analysis[0], no_PCA_analysis[8], no_PCA_analysis[9], label = 'No PCA', color = 'pink')
axs[3,1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], RingSpectrum_analysis[9], label = 'Best Parameters', color = 'pink')
axs[3,1].set_xlabel('SZA')
axs[3,1].set_ylabel('IO dSCD')
axs[3,1].set_title('PCA variation')
    
plt.savefig('IO SCD under different parameter variation categories', dpi = 400)

    
    
#%%



# Calculate a mean and standard deviation for the whole set of retrieval scenarios
# go from the subplots figure not original data importation
# for ease


all_scds = polynom_order_3_wav_428_468_analysis + polynom_order_3_wav_435_465_analysis + polynom_order_3_wav_425_455_analysis \
    + polynom_order_3_wav_410_450_redo_analysis + polynom_order_3_wav_420_460_redo_analysis + polynom_order_3_wav_430_470_redo_analysis \
    + ref_224504_redo_analysis + ref_221505_redo_analysis + ref_223000_redo_analysis \
    + ref_215959_redo_analysis + ref_230002_redo_analysis + ref_210005_redo_analysis \
    + polynomial_order_8_redo_analysis + polynomial_order_7_redo_analysis + polynomial_order_6_redo_analysis \
    + polynom_order_5_with_CLD_sigmas_v2_analysis + polynomial_order_4_redo_analysis + polynom_order_3_with_CLD_sigmas_v2_analysis \
    + polynom_order_2_with_CLD_sigmas_v2_analysis \
    + NO2_294K_Vandaele_ref_redo_analysis + NO2_293K_Burrows_ref_redo_analysis + NO2_293K_Bogumil_ref_redo_analysis \
    + NO2_273K_Burrows_ref_redo_analysis + NO2_273K_Bogumil_ref_redo_analysis + NO2_243K_Bogumil_ref_redo_analysis \
    + NO2_241K_Burrows_ref_redo_analysis + NO2_223K_Bogumil_ref_redo_analysis + NO2_221K_Burrows_ref_redo_analysis \
    + NO2_203K_Bogumi_ref_redo_analysis + NO2_220K_Vandaele_redo_analysis \
    + O3_193K_ref_redo_analysis + O3_203K_ref_redo_analysis + O3_213K_ref_redo_analysis \
    + O3_223K_ref_redo_analysis + O3_233K_ref_redo_analysis + O3_273K_ref_redo_analysis \
    + O3_253K_ref_redo_analysis + O3_263K_ref_redo_analysis \
    + no_shift_redo_analysis + no_stretch_redo_analysis + first_order_stretch_redo_analysis + linear_offset_order_2_redo_analysis \
    + PCA_poly2_analysis +  PCA_poly3_analysis + no_PCA_analysis \
    + RingSpectrum_analysis
                        



num_total_scenarios = 46

mean_scd = all_scds[8] / num_total_scenarios

all_std_devs = np.abs(polynom_order_3_wav_428_468_analysis[8]  - mean_scd) + np.abs(polynom_order_3_wav_435_465_analysis[8] - mean_scd) + np.abs(polynom_order_3_wav_425_455_analysis[8]  - mean_scd) \
    + np.abs(polynom_order_3_wav_410_450_redo_analysis[8]  - mean_scd) + np.abs(polynom_order_3_wav_420_460_redo_analysis[8] - mean_scd)  + np.abs(polynom_order_3_wav_430_470_redo_analysis[8]  - mean_scd) \
    + np.abs(ref_224504_redo_analysis[8] - mean_scd)  + np.abs(ref_221505_redo_analysis[8] - mean_scd)  + np.abs(ref_223000_redo_analysis[8] - mean_scd)  \
    + np.abs(ref_215959_redo_analysis[8] - mean_scd)  + np.abs(ref_230002_redo_analysis[8] - mean_scd)  + np.abs(ref_210005_redo_analysis[8] - mean_scd)  \
    + np.abs(polynomial_order_8_redo_analysis[8] - mean_scd) + np.abs(polynomial_order_7_redo_analysis[8] - mean_scd) + np.abs(polynomial_order_6_redo_analysis[8] - mean_scd) \
    + np.abs(polynom_order_5_with_CLD_sigmas_v2_analysis[8] - mean_scd) + np.abs(polynomial_order_4_redo_analysis[8] - mean_scd) + np.abs(polynom_order_3_with_CLD_sigmas_v2_analysis[8] - mean_scd) \
    + np.abs( polynom_order_2_with_CLD_sigmas_v2_analysis[8] - mean_scd) \
    + np.abs(NO2_294K_Vandaele_ref_redo_analysis[8] - mean_scd) + np.abs( NO2_293K_Burrows_ref_redo_analysis[8] - mean_scd) + np.abs(NO2_293K_Bogumil_ref_redo_analysis[8] - mean_scd) \
    + np.abs(NO2_273K_Burrows_ref_redo_analysis[8] - mean_scd) + np.abs(NO2_273K_Bogumil_ref_redo_analysis[8] - mean_scd) + np.abs(NO2_243K_Bogumil_ref_redo_analysis[8] - mean_scd) \
    + np.abs(NO2_241K_Burrows_ref_redo_analysis[8] - mean_scd) + np.abs(NO2_223K_Bogumil_ref_redo_analysis[8] - mean_scd) + np.abs(NO2_221K_Burrows_ref_redo_analysis[8] - mean_scd) \
    + np.abs(NO2_203K_Bogumi_ref_redo_analysis[8] - mean_scd) + np.abs(NO2_220K_Vandaele_redo_analysis[8] - mean_scd) \
    + np.abs(O3_193K_ref_redo_analysis[8] - mean_scd) + np.abs(O3_203K_ref_redo_analysis[8] - mean_scd) + np.abs(O3_213K_ref_redo_analysis[8] - mean_scd) \
    + np.abs(O3_223K_ref_redo_analysis[8] - mean_scd) + np.abs(O3_233K_ref_redo_analysis[8] - mean_scd) + np.abs(O3_273K_ref_redo_analysis[8] - mean_scd) \
    + np.abs(O3_253K_ref_redo_analysis[8] - mean_scd) + np.abs(O3_263K_ref_redo_analysis[8] - mean_scd) \
    + np.abs(no_shift_redo_analysis[8] - mean_scd) + np.abs(no_stretch_redo_analysis[8] - mean_scd) + np.abs(first_order_stretch_redo_analysis[8] - mean_scd) + np.abs(linear_offset_order_2_redo_analysis[8] - mean_scd) \
    + np.abs(PCA_poly2_analysis[8] - mean_scd) +  np.abs(PCA_poly3_analysis[8] - mean_scd) + np.abs(no_PCA_analysis[8] - mean_scd) \
    + np.abs(RingSpectrum_analysis[8] - mean_scd)



mean_dev = all_std_devs / num_total_scenarios


#%%

plt.errorbar(RingSpectrum_analysis[0], mean_scd, yerr = mean_dev)
plt.title('Mean dSCD IO with retrieval scenario devation')
plt.xlabel('SZA (degrees)')
plt.ylabel('dSCD IO (molecules/cm2)')
plt.grid()
plt.savefig('Mean dSCD IO with retrieval scenario devation', dpi = 400)

#%%


# deviations for each altered parameter


wav_ranges_scds = polynom_order_3_wav_428_468_analysis + polynom_order_3_wav_435_465_analysis + polynom_order_3_wav_425_455_analysis \
    + polynom_order_3_wav_410_450_redo_analysis + polynom_order_3_wav_420_460_redo_analysis + polynom_order_3_wav_430_470_redo_analysis 

wav_ranges_mean_scd = wav_ranges_scds[8] / 6



wavs_std_devs = np.abs(polynom_order_3_wav_428_468_analysis[8]  - wav_ranges_mean_scd) + np.abs(polynom_order_3_wav_435_465_analysis[8] - wav_ranges_mean_scd) + np.abs(polynom_order_3_wav_425_455_analysis[8]  - wav_ranges_mean_scd) \
    + np.abs(polynom_order_3_wav_410_450_redo_analysis[8]  - wav_ranges_mean_scd) + np.abs(polynom_order_3_wav_420_460_redo_analysis[8] - wav_ranges_mean_scd)  + np.abs(polynom_order_3_wav_430_470_redo_analysis[8]  - wav_ranges_mean_scd) 

wavs_mean_dev = wavs_std_devs / 6



refs_scds =  ref_224504_redo_analysis + ref_221505_redo_analysis + ref_223000_redo_analysis \
    + ref_215959_redo_analysis + ref_230002_redo_analysis + ref_210005_redo_analysis 


refs_mean_scd = refs_scds[8] / 6

refs_std_devs = np.abs(ref_224504_redo_analysis[8] - refs_mean_scd)  + np.abs(ref_221505_redo_analysis[8] - refs_mean_scd)  + np.abs(ref_223000_redo_analysis[8] - refs_mean_scd)  \
    + np.abs(ref_215959_redo_analysis[8] - refs_mean_scd)  + np.abs(ref_230002_redo_analysis[8] - refs_mean_scd)  + np.abs(ref_210005_redo_analysis[8] - refs_mean_scd)  


refs_mean_dev = refs_std_devs / 6


polynom_order_scds =   polynomial_order_8_redo_analysis + polynomial_order_7_redo_analysis + polynomial_order_6_redo_analysis \
    + polynom_order_5_with_CLD_sigmas_v2_analysis + polynomial_order_4_redo_analysis + polynom_order_3_with_CLD_sigmas_v2_analysis \
    + polynom_order_2_with_CLD_sigmas_v2_analysis 
    

polynom_orders_mean_scd = polynom_order_scds[8] / 7

polynom_orders_devs =  np.abs(polynomial_order_8_redo_analysis[8] - polynom_orders_mean_scd) + np.abs(polynomial_order_7_redo_analysis[8] - polynom_orders_mean_scd) + np.abs(polynomial_order_6_redo_analysis[8] - polynom_orders_mean_scd) \
    + np.abs(polynom_order_5_with_CLD_sigmas_v2_analysis[8] - polynom_orders_mean_scd) + np.abs(polynomial_order_4_redo_analysis[8] - polynom_orders_mean_scd) + np.abs(polynom_order_3_with_CLD_sigmas_v2_analysis[8] - polynom_orders_mean_scd) \
    + np.abs( polynom_order_2_with_CLD_sigmas_v2_analysis[8] - polynom_orders_mean_scd) 
    
polynom_orders_mean_dev = polynom_orders_devs / 7


NO2_refs_scds =   NO2_294K_Vandaele_ref_redo_analysis + NO2_293K_Burrows_ref_redo_analysis + NO2_293K_Bogumil_ref_redo_analysis \
    + NO2_273K_Burrows_ref_redo_analysis + NO2_273K_Bogumil_ref_redo_analysis + NO2_243K_Bogumil_ref_redo_analysis \
    + NO2_241K_Burrows_ref_redo_analysis + NO2_223K_Bogumil_ref_redo_analysis + NO2_221K_Burrows_ref_redo_analysis \
    + NO2_203K_Bogumi_ref_redo_analysis + NO2_220K_Vandaele_redo_analysis 
    
NO2_refs_mean_scd = NO2_refs_scds[8] / 11


NO2_refs_std_devs =  np.abs(NO2_294K_Vandaele_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs( NO2_293K_Burrows_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs(NO2_293K_Bogumil_ref_redo_analysis[8] - NO2_refs_mean_scd) \
    + np.abs(NO2_273K_Burrows_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs(NO2_273K_Bogumil_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs(NO2_243K_Bogumil_ref_redo_analysis[8] - NO2_refs_mean_scd) \
    + np.abs(NO2_241K_Burrows_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs(NO2_223K_Bogumil_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs(NO2_221K_Burrows_ref_redo_analysis[8] - NO2_refs_mean_scd) \
    + np.abs(NO2_203K_Bogumi_ref_redo_analysis[8] - NO2_refs_mean_scd) + np.abs(NO2_220K_Vandaele_redo_analysis[8] - NO2_refs_mean_scd) 
    

NO2_refs_mean_dev = NO2_refs_std_devs / 11


O3_refs_scds =   O3_193K_ref_redo_analysis + O3_203K_ref_redo_analysis + O3_213K_ref_redo_analysis \
    + O3_223K_ref_redo_analysis + O3_233K_ref_redo_analysis + O3_273K_ref_redo_analysis \
    + O3_253K_ref_redo_analysis + O3_263K_ref_redo_analysis 
    
O3_refs_mean_scd = O3_refs_scds[8] / 8


O3_refs_std_devs =  np.abs(O3_193K_ref_redo_analysis[8] - O3_refs_mean_scd) + np.abs(O3_203K_ref_redo_analysis[8] - O3_refs_mean_scd) + np.abs(O3_213K_ref_redo_analysis[8] - O3_refs_mean_scd) \
    + np.abs(O3_223K_ref_redo_analysis[8] - O3_refs_mean_scd) + np.abs(O3_233K_ref_redo_analysis[8] - O3_refs_mean_scd) + np.abs(O3_273K_ref_redo_analysis[8] - O3_refs_mean_scd) \
    + np.abs(O3_253K_ref_redo_analysis[8] - O3_refs_mean_scd) + np.abs(O3_263K_ref_redo_analysis[8] - O3_refs_mean_scd) 


O3_refs_mean_dev = O3_refs_std_devs / 8


shift_stretch_scds =  no_shift_redo_analysis + no_stretch_redo_analysis + first_order_stretch_redo_analysis + linear_offset_order_2_redo_analysis

mean_shift_stretch_scd = shift_stretch_scds[8] / 4 
    

shift_stretch_std_devs  = np.abs(no_shift_redo_analysis[8] - mean_shift_stretch_scd) + np.abs(no_stretch_redo_analysis[8] - mean_shift_stretch_scd) + np.abs(first_order_stretch_redo_analysis[8] - mean_shift_stretch_scd) + np.abs(linear_offset_order_2_redo_analysis[8] - mean_shift_stretch_scd) 


shift_stretch_mean_dev = shift_stretch_std_devs / 4

PCA_scds =   PCA_poly2_analysis +  PCA_poly3_analysis + no_PCA_analysis 

mean_PCA_scd = PCA_scds[8] / 3


PCA_std_devs = np.abs(PCA_poly2_analysis[8] - mean_PCA_scd) +  np.abs(PCA_poly3_analysis[8] - mean_PCA_scd) + np.abs(no_PCA_analysis[8] - mean_PCA_scd)


PCA_mean_dev = PCA_std_devs / 3

#%%

plt.errorbar(RingSpectrum_analysis[0], wav_ranges_mean_scd, yerr = wavs_mean_dev, label = 'Wavelength Ranges')
plt.errorbar(RingSpectrum_analysis[0], refs_mean_scd, yerr = refs_mean_dev, label = 'Reference Spectra')
plt.errorbar(RingSpectrum_analysis[0], polynom_orders_mean_scd, yerr = polynom_orders_mean_dev, label = 'Polynomial Orders')
plt.errorbar(RingSpectrum_analysis[0], NO2_refs_mean_scd, yerr = NO2_refs_mean_dev, label = 'NO2 Reference Spectra')
plt.errorbar(RingSpectrum_analysis[0], O3_refs_mean_scd, yerr = O3_refs_mean_dev, label = 'O3 Reference Spectra')
plt.errorbar(RingSpectrum_analysis[0], mean_shift_stretch_scd, yerr = shift_stretch_mean_dev, label = 'Shift, Stretch, Offset')
plt.errorbar(RingSpectrum_analysis[0], mean_PCA_scd, yerr = PCA_mean_dev, label = 'Pseudoabsorber Cross Sections')
plt.title('Mean dSCD IO for each varied parameter')
plt.xlabel('SZA (degrees)')
plt.ylabel('dSCD IO (molecules/cm2)')
plt.grid()
plt.legend(loc = 'upper left') 
plt.savefig('Mean dSCD IO by retrieval scenario with devation', dpi = 400)

