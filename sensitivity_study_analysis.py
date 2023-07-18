# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:50:09 2023

@author: lme19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


#%%

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
        
   
    
   
#%%

best_params_with_IO_run2_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\best_params_with_IO_run2.ASC', '2022-08-23 22:00:05')
polynom_order_3_wav_428_468_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynom_order_3_wav_428_468.ASC', '2022-08-23 22:00:05')
polynomial_order_8_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_8_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_7_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_7_redo.ASC', '2022-08-23 22:00:05')
polynomial_order_6_redo_analysis = clean_data_high(r'C:\Users\lme19\Documents\Test_data\polynomial_order_6_redo.ASC', '2022-08-23 22:00:05')
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

#Wavelength ranges high SZAs


plt.rcParams.update({'font.size': 5, 'errorbar.capsize' : 0, 'lines.markersize' : 0.8, 'lines.marker' : '.', 'lines.linewidth': 0.2})

plt.errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[8], yerr = polynom_order_3_wav_428_468_analysis[9], label = '428-468', linewidth = 0, elinewidth = 0.1, marker = '.', capsize = 0, markersize = 0.5)
plt.errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[8], yerr = polynom_order_3_wav_435_465_analysis[9], label = '435-465')
plt.errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[8], yerr = polynom_order_3_wav_425_455_analysis[9], label = '425-455')
plt.errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[8], yerr = polynom_order_3_wav_410_450_redo_analysis[9], label = '410-450')
plt.errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[8], yerr = polynom_order_3_wav_420_460_redo_analysis[9], label = '420-460')
plt.errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[8], yerr = polynom_order_3_wav_430_470_redo_analysis[9], label = '430-470')
plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9], label = '425-465')

plt.title('IO SCDs for Different Wavelength Fit Ranges (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[6], yerr = polynom_order_3_wav_428_468_analysis[7],  label = 'polynom_order_3_wav_428_468')
plt.errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[6], yerr = polynom_order_3_wav_435_465_analysis[7], label = 'polynom_order_3_wav_435_465')
plt.errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[6], yerr = polynom_order_3_wav_425_455_analysis[7], label = 'polynom_order_3_wav_425_455')
plt.errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[6], yerr = polynom_order_3_wav_410_450_redo_analysis[7], label = 'polynom_order_3_wav_410_450')
plt.errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[6], yerr = polynom_order_3_wav_420_460_redo_analysis[7], label = 'polynom_order_3_wav_420_460')
plt.errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[6], yerr = polynom_order_3_wav_430_470_redo_analysis[7], label = 'polynom_order_3_wav_430_470')
plt.title('CLD for Different Wavelength Fit Ranges (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


#no errorbars on the RMS
    
plt.plot(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[5], label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[5], label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[5], label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[5], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[5], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[5], label = 'polynom_order_3_wav_430_470')
plt.title('RMS for Different Wavelength Fit Ranges (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#CLD plot without errors since thez mostly overpower the differences between the CLD graphs


plt.plot(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[6],  label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[6],  label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[6],  label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[6], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[6], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[6],  label = 'polynom_order_3_wav_430_470')
plt.title('CLD for Different Wavelength Fit Ranges (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

    
#%%

#Wavelength ranges low SZAs

plt.errorbar(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[8], yerr = polynom_order_3_wav_428_468_low[9], label = 'polynom_order_3_wav_428_468')
plt.errorbar(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[8], yerr = polynom_order_3_wav_435_465_low[9], label = 'polynom_order_3_wav_435_465')
plt.errorbar(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[8], yerr = polynom_order_3_wav_425_455_low[9], label = 'polynom_order_3_wav_425_455')
plt.errorbar(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[8], yerr = polynom_order_3_wav_410_450_redo_low[9], label = 'polynom_order_3_wav_410_450')
plt.errorbar(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[8], yerr = polynom_order_3_wav_420_460_redo_low[9], label = 'polynom_order_3_wav_420_460')
plt.errorbar(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[8], yerr = polynom_order_3_wav_430_470_redo_low[9], label = 'polynom_order_3_wav_430_470')
plt.title('IO SCDs for Different Wavelength Fit Ranges (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[6], yerr = polynom_order_3_wav_428_468_low[7], label = 'polynom_order_3_wav_428_468')
plt.errorbar(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[6], yerr =  polynom_order_3_wav_435_465_low[7], label = 'polynom_order_3_wav_435_465')
plt.errorbar(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[6], yerr = polynom_order_3_wav_425_455_low[7], label = 'polynom_order_3_wav_425_455')
plt.errorbar(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[6], yerr = polynom_order_3_wav_410_450_redo_low[7], label = 'polynom_order_3_wav_410_450')
plt.errorbar(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[6], yerr = polynom_order_3_wav_420_460_redo_low[7], label = 'polynom_order_3_wav_420_460')
plt.errorbar(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[6], yerr = polynom_order_3_wav_430_470_redo_low[7], label = 'polynom_order_3_wav_430_470')
plt.title('CLD for Different Wavelength Fit Ranges (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

    
plt.plot(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[5], label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[5], label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[5], label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[5], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[5], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[5], label = 'polynom_order_3_wav_430_470')
plt.title('RMS for Different Wavelength Fit Ranges (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 


plt.plot(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[6], label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[6], label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[6], label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[6], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[6], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[6], label = 'polynom_order_3_wav_430_470')
plt.title('CLD for Different Wavelength Fit Ranges (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

#%%


#reference spectra High SZAs

plt.errorbar(ref_224504_redo_analysis[0], ref_224504_redo_analysis[8], yerr = ref_224504_redo_analysis[9], label = 'ref_224504')
plt.errorbar(ref_221505_redo_analysis[0], ref_221505_redo_analysis[8], yerr = ref_221505_redo_analysis[9], label = 'ref_221505')
plt.errorbar(ref_223000_redo_analysis[0], ref_223000_redo_analysis[8], yerr = ref_223000_redo_analysis[9], label = 'ref_223000')
plt.errorbar(ref_215959_redo_analysis[0], ref_215959_redo_analysis[8], yerr = ref_215959_redo_analysis[9], label = 'ref_215959')
plt.errorbar(ref_230002_redo_analysis[0], ref_230002_redo_analysis[8], yerr = ref_230002_redo_analysis[9], label = 'ref_230002')
plt.errorbar(ref_210005_redo_analysis[0], ref_210005_redo_analysis[8], yerr = ref_210005_redo_analysis[9], label = 'ref_210005')
plt.title('IO SCDs for Different Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(ref_224504_redo_analysis[0], ref_224504_redo_analysis[6], yerr = ref_224504_redo_analysis[7], label = 'ref_224504')
plt.errorbar(ref_221505_redo_analysis[0], ref_221505_redo_analysis[6], yerr = ref_221505_redo_analysis[7], label = 'ref_221505')
plt.errorbar(ref_223000_redo_analysis[0], ref_223000_redo_analysis[6], yerr = ref_223000_redo_analysis[7], label = 'ref_223000')
plt.errorbar(ref_215959_redo_analysis[0], ref_215959_redo_analysis[6], yerr = ref_215959_redo_analysis[7], label = 'ref_215959')
plt.errorbar(ref_230002_redo_analysis[0], ref_230002_redo_analysis[6], yerr = ref_230002_redo_analysis[7], label = 'ref_230002')
plt.errorbar(ref_210005_redo_analysis[0], ref_210005_redo_analysis[6], yerr = ref_210005_redo_analysis[7], label = 'ref_210005')
plt.title('CLD for Different Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(ref_224504_redo_analysis[0], ref_224504_redo_analysis[5], label = 'ref_224504')
plt.plot(ref_221505_redo_analysis[0], ref_221505_redo_analysis[5], label = 'ref_221505')
plt.plot(ref_223000_redo_analysis[0], ref_223000_redo_analysis[5], label = 'ref_223000')
plt.plot(ref_215959_redo_analysis[0], ref_215959_redo_analysis[5], label = 'ref_215959')
plt.plot(ref_230002_redo_analysis[0], ref_230002_redo_analysis[5], label = 'ref_230002')
plt.plot(ref_210005_redo_analysis[0], ref_210005_redo_analysis[5], label = 'ref_210005')
plt.title('RMS for Different Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%


#reference spectra Low SZAs

plt.errorbar(ref_224504_redo_low[0], ref_224504_redo_low[8], yerr = ref_224504_redo_low[9], label = 'ref_224504')
plt.errorbar(ref_221505_redo_low[0], ref_221505_redo_low[8], yerr = ref_221505_redo_low[9], label = 'ref_221505')
plt.errorbar(ref_223000_redo_low[0], ref_223000_redo_low[8], yerr = ref_223000_redo_low[9], label = 'ref_223000')
plt.errorbar(ref_215959_redo_low[0], ref_215959_redo_low[8], yerr = ref_215959_redo_low[9], label = 'ref_215959')
plt.errorbar(ref_230002_redo_low[0], ref_230002_redo_low[8], yerr = ref_230002_redo_low[9], label = 'ref_230002')
plt.errorbar(ref_210005_redo_low[0], ref_210005_redo_low[8], yerr = ref_210005_redo_low[9], label = 'ref_210005')
plt.title('IO SCDs for Different Reference Spectra (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(ref_224504_redo_low[0], ref_224504_redo_low[6], yerr = ref_224504_redo_low[7], label = 'ref_224504')
plt.errorbar(ref_221505_redo_low[0], ref_221505_redo_low[6], yerr = ref_221505_redo_low[7], label = 'ref_221505')
plt.errorbar(ref_223000_redo_low[0], ref_223000_redo_low[6], yerr = ref_223000_redo_low[7], label = 'ref_223000')
plt.errorbar(ref_215959_redo_low[0], ref_215959_redo_low[6], yerr = ref_215959_redo_low[7], label = 'ref_215959')
plt.errorbar(ref_230002_redo_low[0], ref_230002_redo_low[6], yerr = ref_230002_redo_low[7], label = 'ref_230002')
plt.errorbar(ref_210005_redo_low[0], ref_210005_redo_low[6], yerr = ref_210005_redo_low[7], label = 'ref_210005')
plt.title('CLD for Different Reference Spectra (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(ref_224504_redo_low[0], ref_224504_redo_low[5], label = 'ref_224504')
plt.plot(ref_221505_redo_low[0], ref_221505_redo_low[5], label = 'ref_221505')
plt.plot(ref_223000_redo_low[0], ref_223000_redo_low[5], label = 'ref_223000')
plt.plot(ref_215959_redo_low[0], ref_215959_redo_low[5], label = 'ref_215959')
plt.plot(ref_230002_redo_low[0], ref_230002_redo_low[5], label = 'ref_230002')
plt.plot(ref_210005_redo_low[0], ref_210005_redo_low[5], label = 'ref_210005')
plt.title('RMS for Different Reference Spectra (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%

#polynomial orders for high SZAs


plt.errorbar(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[8], yerr = polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 8')
plt.errorbar(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[8], yerr = polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 7')
plt.errorbar(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[8], yerr = polynomial_order_8_redo_analysis[9], label = 'Polynomial Order 6')
plt.errorbar(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[8], yerr = polynom_order_5_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 5')
plt.errorbar(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[8], yerr = polynomial_order_4_redo_analysis[9], label = 'Polynomial Order 4')
plt.errorbar(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[8], yerr = polynom_order_3_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 3')
plt.errorbar(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[8], yerr = polynom_order_2_with_CLD_sigmas_v2_analysis[9], label = 'Polynomial Order 2')
plt.title('IO SCDs for Different Order Polynomials (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[6], yerr = polynomial_order_8_redo_analysis[7], label = 'Polynomial Order 8')
plt.errorbar(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[6], yerr = polynomial_order_8_redo_analysis[7], label = 'Polynomial Order 7')
plt.errorbar(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[6], yerr = polynomial_order_8_redo_analysis[7], label = 'Polynomial Order 6')
plt.errorbar(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[6], yerr = polynom_order_5_with_CLD_sigmas_v2_analysis[7], label = 'Polynomial Order 5')
plt.errorbar(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[6], yerr = polynomial_order_4_redo_analysis[7], label = 'Polynomial Order 4')
plt.errorbar(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[6], yerr = polynom_order_3_with_CLD_sigmas_v2_analysis[7], label = 'Polynomial Order 3')
plt.errorbar(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[6], yerr = polynom_order_2_with_CLD_sigmas_v2_analysis[7], label = 'Polynomial Order 2')
plt.title('CLD for Different Order Polynomials (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 8')
plt.plot(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 7')
plt.plot(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 6')
plt.plot(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 5')
plt.plot(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[5], label = 'Polynomial Order 4')
plt.plot(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 3')
plt.plot(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 2')
plt.title('RMS for Different Order Polynomials (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%

#polynomial orders low SZA


plt.errorbar(polynomial_order_8_redo_low[0], polynomial_order_8_redo_low[8], yerr = polynomial_order_8_redo_low[9], label = 'Polynomial Order 7')
plt.errorbar(polynomial_order_6_redo_low[0], polynomial_order_8_redo_low[8], yerr = polynomial_order_8_redo_low[9], label = 'Polynomial Order 6')
plt.errorbar(polynom_order_5_with_CLD_sigmas_v2_low[0], polynom_order_5_with_CLD_sigmas_v2_low[8], yerr = polynom_order_5_with_CLD_sigmas_v2_low[9], label = 'Polynomial Order 5')
plt.errorbar(polynomial_order_4_redo_low[0], polynomial_order_4_redo_low[8], yerr = polynomial_order_4_redo_low[9], label = 'Polynomial Order 4')
plt.errorbar(polynom_order_3_with_CLD_sigmas_v2_low[0], polynom_order_3_with_CLD_sigmas_v2_low[8], yerr = polynom_order_3_with_CLD_sigmas_v2_low[9], label = 'Polynomial Order 3')
plt.errorbar(polynom_order_2_with_CLD_sigmas_v2_low[0], polynom_order_2_with_CLD_sigmas_v2_low[8], yerr = polynom_order_2_with_CLD_sigmas_v2_low[9], label = 'Polynomial Order 2')
plt.title('IO SCDs for Different Order Polynomials (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(polynomial_order_8_redo_low[0], polynomial_order_8_redo_low[6], yerr = polynomial_order_8_redo_low[7], label = 'Polynomial Order 8')
plt.errorbar(polynomial_order_7_redo_low[0], polynomial_order_8_redo_low[6], yerr = polynomial_order_8_redo_low[7], label = 'Polynomial Order 7')
plt.errorbar(polynomial_order_6_redo_low[0], polynomial_order_8_redo_low[6], yerr = polynomial_order_8_redo_low[7], label = 'Polynomial Order 6')
plt.errorbar(polynom_order_5_with_CLD_sigmas_v2_low[0], polynom_order_5_with_CLD_sigmas_v2_low[6], yerr = polynom_order_5_with_CLD_sigmas_v2_low[7], label = 'Polynomial Order 5')
plt.errorbar(polynomial_order_4_redo_low[0], polynomial_order_4_redo_low[6], yerr = polynomial_order_4_redo_low[7], label = 'Polynomial Order 4')
plt.errorbar(polynom_order_3_with_CLD_sigmas_v2_low[0], polynom_order_3_with_CLD_sigmas_v2_low[6], yerr = polynom_order_3_with_CLD_sigmas_v2_low[7], label = 'Polynomial Order 3')
plt.errorbar(polynom_order_2_with_CLD_sigmas_v2_low[0], polynom_order_2_with_CLD_sigmas_v2_low[6], yerr = polynom_order_2_with_CLD_sigmas_v2_low[7], label = 'Polynomial Order 2')
plt.title('CLD for Different Order Polynomials (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(polynomial_order_8_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 8')
plt.plot(polynomial_order_7_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 7')
plt.plot(polynomial_order_6_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 6')
plt.plot(polynom_order_5_with_CLD_sigmas_v2_low[0], polynom_order_5_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 5')
plt.plot(polynomial_order_4_redo_low[0], polynomial_order_4_redo_low[5], label = 'Polynomial Order 4')
plt.plot(polynom_order_3_with_CLD_sigmas_v2_low[0], polynom_order_3_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 3')
plt.plot(polynom_order_2_with_CLD_sigmas_v2_low[0], polynom_order_2_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 2')
plt.title('RMS for Different Order Polynomials (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%


#NO2 reference spectra high SZAs

plt.errorbar(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[8], yerr = NO2_294K_Vandaele_ref_redo_analysis[9], label = 'NO2_294K_Vandaele')
plt.errorbar(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[8], yerr = NO2_293K_Burrows_ref_redo_analysis[9], label = 'NO2_293K_Burrows')
plt.errorbar(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[8], yerr = NO2_293K_Bogumil_ref_redo_analysis[9], label = 'NO2_293K_Bogumil')
plt.errorbar(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[8], yerr = NO2_273K_Burrows_ref_redo_analysis[9], label = 'NO2_273K_Burrows')
plt.errorbar(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[8], yerr = NO2_273K_Bogumil_ref_redo_analysis[9], label = 'NO2_273K_Bogumil')
plt.errorbar(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[8], yerr = NO2_243K_Bogumil_ref_redo_analysis[9], label = 'NO2_243K_Bogumil')
plt.errorbar(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[8], yerr = NO2_241K_Burrows_ref_redo_analysis[9], label = 'NO2_241K_Burrows')
plt.errorbar(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[8], yerr = NO2_223K_Bogumil_ref_redo_analysis[9], label = 'NO2_223K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8], yerr = NO2_221K_Burrows_ref_redo_analysis[9], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[8], yerr = NO2_203K_Bogumi_ref_redo_analysis[9], label = 'NO2_203K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8], yerr = NO2_221K_Burrows_ref_redo_analysis[9], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[8], yerr = NO2_220K_Vandaele_redo_analysis[9], label = 'NO2_220K_Vandaele')
plt.title('IO SCDs for Different NO2 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

plt.errorbar(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[6], yerr = NO2_294K_Vandaele_ref_redo_analysis[7], label = 'NO2_294K_Vandaele')
plt.errorbar(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[6], yerr = NO2_293K_Burrows_ref_redo_analysis[7], label = 'NO2_293K_Burrows')
plt.errorbar(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[6], yerr = NO2_293K_Bogumil_ref_redo_analysis[7], label = 'NO2_293K_Bogumil')
plt.errorbar(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[6], yerr = NO2_273K_Burrows_ref_redo_analysis[7], label = 'NO2_273K_Burrows')
plt.errorbar(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[6], yerr = NO2_273K_Bogumil_ref_redo_analysis[7], label = 'NO2_273K_Bogumil')
plt.errorbar(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[6], yerr = NO2_243K_Bogumil_ref_redo_analysis[7], label = 'NO2_243K_Bogumil')
plt.errorbar(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[6], yerr = NO2_241K_Burrows_ref_redo_analysis[7], label = 'NO2_241K_Burrows')
plt.errorbar(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[6], yerr = NO2_223K_Bogumil_ref_redo_analysis[7], label = 'NO2_223K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[6], yerr = NO2_221K_Burrows_ref_redo_analysis[7], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[6], yerr = NO2_203K_Bogumi_ref_redo_analysis[7], label = 'NO2_203K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[6], yerr = NO2_221K_Burrows_ref_redo_analysis[7], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[6], yerr = NO2_220K_Vandaele_redo_analysis[7], label = 'NO2_220K_Vandaele')
plt.title('CLD for Different NO2 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

plt.plot(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[5], label = 'NO2_294K_Vandaele')
plt.plot(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[5], label = 'NO2_293K_Burrows')
plt.plot(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[5], label = 'NO2_293K_Bogumil')
plt.plot(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[5], label = 'NO2_273K_Burrows')
plt.plot(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[5], label = 'NO2_273K_Bogumil')
plt.plot(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[5], label = 'NO2_243K_Bogumil')
plt.plot(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[5], label = 'NO2_241K_Burrows')
plt.plot(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[5], label = 'NO2_223K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[5], label = 'NO2_203K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[5], label = 'NO2_220K_Vandaele')
plt.title('RMS for Different NO2 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%

#NO2 reference spectra low SZAs

plt.errorbar(NO2_294K_Vandaele_ref_redo_low[0], NO2_294K_Vandaele_ref_redo_low[8], yerr = NO2_294K_Vandaele_ref_redo_low[9], label = 'NO2_294K_Vandaele')
plt.errorbar(NO2_293K_Burrows_ref_redo_low[0], NO2_293K_Burrows_ref_redo_low[8], yerr = NO2_293K_Burrows_ref_redo_low[9], label = 'NO2_293K_Burrows')
plt.errorbar(NO2_293K_Bogumil_ref_redo_low[0], NO2_293K_Bogumil_ref_redo_low[8], yerr = NO2_293K_Bogumil_ref_redo_low[9], label = 'NO2_293K_Bogumil')
plt.errorbar(NO2_273K_Burrows_ref_redo_low[0], NO2_273K_Burrows_ref_redo_low[8], yerr = NO2_273K_Burrows_ref_redo_low[9], label = 'NO2_273K_Burrows')
plt.errorbar(NO2_273K_Bogumil_ref_redo_low[0], NO2_273K_Bogumil_ref_redo_low[8], yerr = NO2_273K_Bogumil_ref_redo_low[9], label = 'NO2_273K_Bogumil')
plt.errorbar(NO2_243K_Bogumil_ref_redo_low[0], NO2_243K_Bogumil_ref_redo_low[8], yerr = NO2_243K_Bogumil_ref_redo_low[9], label = 'NO2_243K_Bogumil')
plt.errorbar(NO2_241K_Burrows_ref_redo_low[0], NO2_241K_Burrows_ref_redo_low[8], yerr = NO2_241K_Burrows_ref_redo_low[9], label = 'NO2_241K_Burrows')
plt.errorbar(NO2_223K_Bogumil_ref_redo_low[0], NO2_223K_Bogumil_ref_redo_low[8], yerr = NO2_223K_Bogumil_ref_redo_low[9], label = 'NO2_223K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[8], yerr = NO2_221K_Burrows_ref_redo_low[9], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_203K_Bogumi_ref_redo_low[0], NO2_203K_Bogumi_ref_redo_low[8], yerr = NO2_203K_Bogumi_ref_redo_low[9], label = 'NO2_203K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[8], yerr = NO2_221K_Burrows_ref_redo_low[9], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_220K_Vandaele_redo_low[0], NO2_220K_Vandaele_redo_low[8], yerr = NO2_220K_Vandaele_redo_low[9], label = 'NO2_220K_Vandaele')
plt.title('IO SCDs for Different NO2 Reference Spectra (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

plt.errorbar(NO2_294K_Vandaele_ref_redo_low[0], NO2_294K_Vandaele_ref_redo_low[6], yerr = NO2_294K_Vandaele_ref_redo_low[7], label = 'NO2_294K_Vandaele')
plt.errorbar(NO2_293K_Burrows_ref_redo_low[0], NO2_293K_Burrows_ref_redo_low[6], yerr = NO2_293K_Burrows_ref_redo_low[7], label = 'NO2_293K_Burrows')
plt.errorbar(NO2_293K_Bogumil_ref_redo_low[0], NO2_293K_Bogumil_ref_redo_low[6], yerr = NO2_293K_Bogumil_ref_redo_low[7], label = 'NO2_293K_Bogumil')
plt.errorbar(NO2_273K_Burrows_ref_redo_low[0], NO2_273K_Burrows_ref_redo_low[6], yerr = NO2_273K_Burrows_ref_redo_low[7], label = 'NO2_273K_Burrows')
plt.errorbar(NO2_273K_Bogumil_ref_redo_low[0], NO2_273K_Bogumil_ref_redo_low[6], yerr = NO2_273K_Bogumil_ref_redo_low[7], label = 'NO2_273K_Bogumil')
plt.errorbar(NO2_243K_Bogumil_ref_redo_low[0], NO2_243K_Bogumil_ref_redo_low[6], yerr = NO2_243K_Bogumil_ref_redo_low[7], label = 'NO2_243K_Bogumil')
plt.errorbar(NO2_241K_Burrows_ref_redo_low[0], NO2_241K_Burrows_ref_redo_low[6], yerr = NO2_241K_Burrows_ref_redo_low[7], label = 'NO2_241K_Burrows')
plt.errorbar(NO2_223K_Bogumil_ref_redo_low[0], NO2_223K_Bogumil_ref_redo_low[6], yerr = NO2_223K_Bogumil_ref_redo_low[7], label = 'NO2_223K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[6], yerr = NO2_221K_Burrows_ref_redo_low[7], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_203K_Bogumi_ref_redo_low[0], NO2_203K_Bogumi_ref_redo_low[6], yerr = NO2_203K_Bogumi_ref_redo_low[7], label = 'NO2_203K_Bogumil')
plt.errorbar(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[6], yerr = NO2_221K_Burrows_ref_redo_low[7], label = 'NO2_221K_Burrows')
plt.errorbar(NO2_220K_Vandaele_redo_low[0], NO2_220K_Vandaele_redo_low[6], yerr = NO2_220K_Vandaele_redo_low[7], label = 'NO2_220K_Vandaele')
plt.title('CLD for Different NO2 Reference Spectra (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 

plt.plot(NO2_294K_Vandaele_ref_redo_low[0], NO2_294K_Vandaele_ref_redo_low[5], label = 'NO2_294K_Vandaele')
plt.plot(NO2_293K_Burrows_ref_redo_low[0], NO2_293K_Burrows_ref_redo_low[5], label = 'NO2_293K_Burrows')
plt.plot(NO2_293K_Bogumil_ref_redo_low[0], NO2_293K_Bogumil_ref_redo_low[5], label = 'NO2_293K_Bogumil')
plt.plot(NO2_273K_Burrows_ref_redo_low[0], NO2_273K_Burrows_ref_redo_low[5], label = 'NO2_273K_Burrows')
plt.plot(NO2_273K_Bogumil_ref_redo_low[0], NO2_273K_Bogumil_ref_redo_low[5], label = 'NO2_273K_Bogumil')
plt.plot(NO2_243K_Bogumil_ref_redo_low[0], NO2_243K_Bogumil_ref_redo_low[5], label = 'NO2_243K_Bogumil')
plt.plot(NO2_241K_Burrows_ref_redo_low[0], NO2_241K_Burrows_ref_redo_low[5], label = 'NO2_241K_Burrows')
plt.plot(NO2_223K_Bogumil_ref_redo_low[0], NO2_223K_Bogumil_ref_redo_low[5], label = 'NO2_223K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_203K_Bogumi_ref_redo_low[0], NO2_203K_Bogumi_ref_redo_low[5], label = 'NO2_203K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_220K_Vandaele_redo_low[0], NO2_220K_Vandaele_redo_low[5], label = 'NO2_220K_Vandaele')
plt.title('RMS for Different NO2 Reference Spectra (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%

#O3 reference spectra high SZAs


plt.errorbar(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[8], yerr = O3_193K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[8], yerr = O3_203K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[8], yerr = O3_213K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[8], yerr = O3_223K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[8], yerr = O3_233K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[8], yerr = O3_273K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[8], yerr = O3_253K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.errorbar(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[8], yerr = O3_263K_ref_redo_analysis[9], label = 'O3 ref 193K')
plt.title('IO SCDs for Different O3 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[6], yerr = O3_193K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[6], yerr = O3_203K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[6], yerr = O3_213K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[6], yerr = O3_223K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[6], yerr = O3_233K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[6], yerr = O3_273K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[6], yerr = O3_253K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.errorbar(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[6], yerr = O3_263K_ref_redo_analysis[7], label = 'O3 ref 193K')
plt.title('CLD for Different O3 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.title('RMS for Different O3 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%

#O3 reference spectra low SZAS

plt.errorbar(O3_193K_ref_redo_low[0], O3_193K_ref_redo_low[8], yerr = O3_193K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_203K_ref_redo_low[0], O3_203K_ref_redo_low[8], yerr = O3_203K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_213K_ref_redo_low[0], O3_213K_ref_redo_low[8], yerr = O3_213K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_223K_ref_redo_low[0], O3_223K_ref_redo_low[8], yerr = O3_223K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_233K_ref_redo_low[0], O3_233K_ref_redo_low[8], yerr = O3_233K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_273K_ref_redo_low[0], O3_273K_ref_redo_low[8], yerr = O3_273K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_253K_ref_redo_low[0], O3_253K_ref_redo_low[8], yerr = O3_253K_ref_redo_low[9], label = 'O3 ref 193K')
plt.errorbar(O3_263K_ref_redo_low[0], O3_263K_ref_redo_low[8], yerr = O3_263K_ref_redo_low[9], label = 'O3 ref 193K')
plt.title('IO SCDs for Different O3 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(O3_193K_ref_redo_low[0], O3_193K_ref_redo_low[6], yerr = O3_193K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_203K_ref_redo_low[0], O3_203K_ref_redo_low[6], yerr = O3_203K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_213K_ref_redo_low[0], O3_213K_ref_redo_low[6], yerr = O3_213K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_223K_ref_redo_low[0], O3_223K_ref_redo_low[6], yerr = O3_223K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_233K_ref_redo_low[0], O3_233K_ref_redo_low[6], yerr = O3_233K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_273K_ref_redo_low[0], O3_273K_ref_redo_low[6], yerr = O3_273K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_253K_ref_redo_low[0], O3_253K_ref_redo_low[6], yerr = O3_253K_ref_redo_low[7], label = 'O3 ref 193K')
plt.errorbar(O3_263K_ref_redo_low[0], O3_263K_ref_redo_low[6], yerr = O3_263K_ref_redo_low[7], label = 'O3 ref 193K')
plt.title('CLD for Different O3 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(O3_193K_ref_redo_low[0], O3_193K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_203K_ref_redo_low[0], O3_203K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_213K_ref_redo_low[0], O3_213K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_223K_ref_redo_low[0], O3_223K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_233K_ref_redo_low[0], O3_233K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_273K_ref_redo_low[0], O3_273K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_253K_ref_redo_low[0], O3_253K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_263K_ref_redo_low[0], O3_263K_ref_redo_low[5], label = 'O3 ref 193K')
plt.title('RMS for Different O3 Reference Spectra (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 

#%%

#Miscellaneous high SZA

plt.errorbar(no_shift_redo_analysis[0], no_shift_redo_analysis[8], yerr =  no_shift_redo_analysis[9])
plt.errorbar(no_stretch_redo_analysis[0], no_stretch_redo_analysis[8], yerr = no_stretch_redo_analysis[9])
plt.errorbar(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[8], yerr = first_order_stretch_redo_analysis[9])
plt.errorbar(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[8], yerr = linear_offset_order_2_redo_analysis[9])
plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9] , label = 'Best Parameters')
plt.title('IO SCDs for Different Shift, Stretch and Offset (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(no_shift_redo_analysis[0], no_shift_redo_analysis[6], yerr =  no_shift_redo_analysis[7])
plt.errorbar(no_stretch_redo_analysis[0], no_stretch_redo_analysis[6], yerr = no_stretch_redo_analysis[7])
plt.errorbar(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[6], yerr = first_order_stretch_redo_analysis[7])
plt.errorbar(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[6], yerr = linear_offset_order_2_redo_analysis[7])
plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[6], yerr = best_params_with_IO_run2_analysis[7] , label = 'Best Parameters')
plt.title('CLD for Different Shift, Stretch and Offset (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(no_shift_redo_analysis[0], no_shift_redo_analysis[5])
plt.plot(no_stretch_redo_analysis[0], no_stretch_redo_analysis[5])
plt.plot(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[5])
plt.plot(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[5])
plt.title('RMS for Different Shift, Stretch and Offset (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 


#%%

#Miscellaneous low SZA

plt.errorbar(no_shift_redo_low[0], no_shift_redo_low[8], yerr =  no_shift_redo_low[9], label = 'No shift')
plt.errorbar(no_stretch_redo_low[0], no_stretch_redo_low[8], yerr = no_stretch_redo_low[9], label = 'No Stretch')
plt.errorbar(first_order_stretch_redo_low[0], first_order_stretch_redo_low[8], yerr = first_order_stretch_redo_low[9], label = 'First Order Stretch')
plt.errorbar(linear_offset_order_2_redo_low[0], linear_offset_order_2_redo_low[8], yerr = linear_offset_order_2_redo_low[9], label = '2nd Order Linear Offset')
plt.errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9] , label = 'Best Parameters')
plt.title('IO SCDs for Different Shift, Stretch and Offset (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(no_shift_redo_low[0], no_shift_redo_low[6], yerr =  no_shift_redo_low[7], label = 'No shift')
plt.errorbar(no_stretch_redo_low[0], no_stretch_redo_low[6], yerr = no_stretch_redo_low[7], label = 'No Stretch')
plt.errorbar(first_order_stretch_redo_low[0], first_order_stretch_redo_low[6], yerr = first_order_stretch_redo_low[7], label = 'First Order Stretch')
plt.errorbar(linear_offset_order_2_redo_low[0], linear_offset_order_2_redo_low[6], yerr = linear_offset_order_2_redo_low[7], label = '2nd Order Linear Offset')
plt.errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[6], yerr = best_params_with_IO_run2_low[7] , label = 'Best Parameters')
plt.title('CLD for Different Shift, Stretch and Offset (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.plot(no_shift_redo_low[0], no_shift_redo_low[5], label = 'No shift')
plt.plot(no_stretch_redo_low[0], no_stretch_redo_low[5], label = 'No Stretch')
plt.plot(first_order_stretch_redo_low[0], first_order_stretch_redo_low[5], label = 'First Order Stretch')
plt.plot(linear_offset_order_2_redo_low[0], linear_offset_order_2_redo_low[5], label = '2nd Order Linear Offset')
plt.plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = 'Best Parameters')
plt.title('RMS for Different Shift, Stretch and Offset (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_RMS')
plt.legend() 
plt.show() 


#%%

# Different PCA cross sections high SZA

plt.errorbar(PCA_poly2_analysis[0], PCA_poly2_analysis[8], yerr =  PCA_poly2_analysis[9], label = 'Poly2')
plt.errorbar(PCA_poly3_analysis[0], PCA_poly3_analysis[8], yerr =  PCA_poly3_analysis[9], label = 'Poly3')
plt.errorbar(no_PCA_analysis[0], no_PCA_analysis[8], yerr =  no_PCA_analysis[9], label = 'No PCA')
plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9] , label = 'Best Parameters')
plt.title('IO SCDs for Different Pseudoabsorber Cross Sections (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(PCA_poly2_analysis[0], PCA_poly2_analysis[6], yerr =  PCA_poly2_analysis[7], label = 'Poly2')
plt.errorbar(PCA_poly3_analysis[0], PCA_poly3_analysis[6], yerr =  PCA_poly3_analysis[7], label = 'Poly3')
plt.errorbar(no_PCA_analysis[0], no_PCA_analysis[6], yerr =  no_PCA_analysis[7], label = 'No PCA')
plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[6], yerr = best_params_with_IO_run2_analysis[7] , label = 'Best Parameters')
plt.title('CLD for Different Pseudoabsorber Cross Sections (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('CLD')
plt.legend() 
plt.show() 


plt.plot(PCA_poly2_analysis[0], PCA_poly2_analysis[5], label = 'Poly2')
plt.plot(PCA_poly3_analysis[0], PCA_poly3_analysis[5],  label = 'Poly3')
plt.plot(no_PCA_analysis[0], no_PCA_analysis[5], label = 'No PCA')
plt.plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5], label = 'Best Parameters')
plt.title('RMS for Different Pseudoabsorber Cross Sections(High SZAs)')
plt.xlabel('SZA')
plt.ylabel('RMS')
plt.legend() 
plt.show() 

#%%

#Different PCA cross sections Low SZA

plt.errorbar(PCA_poly2_low[0], PCA_poly2_low[8], yerr =  PCA_poly2_low[9], label = 'Poly2')
plt.errorbar(PCA_poly3_low[0], PCA_poly3_low[8], yerr =  PCA_poly3_low[9], label = 'Poly3')
plt.errorbar(no_PCA_low[0], no_PCA_low[8], yerr =  no_PCA_low[9], label = 'No PCA')
plt.errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9] , label = 'Best Parameters')
plt.title('IO SCDs for Different Pseudoabsorber Cross Sections (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show() 


plt.errorbar(PCA_poly2_low[0], PCA_poly2_low[6], yerr =  PCA_poly2_low[7], label = 'Poly2')
plt.errorbar(PCA_poly3_low[0], PCA_poly3_low[6], yerr =  PCA_poly3_low[7], label = 'Poly3')
plt.errorbar(no_PCA_low[0], no_PCA_low[6], yerr =  no_PCA_low[7], label = 'No PCA')
plt.errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[6], yerr = best_params_with_IO_run2_low[7] , label = 'Best Parameters')
plt.title('CLD for Different Pseudoabsorber Cross Sections (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('CLD')
plt.legend() 
plt.show() 


plt.plot(PCA_poly2_low[0], PCA_poly2_low[5], label = 'Poly2')
plt.plot(PCA_poly3_low[0], PCA_poly3_low[5],  label = 'Poly3')
plt.plot(no_PCA_low[0], no_PCA_low[5], label = 'No PCA')
plt.plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = 'Best Parameters')
plt.title('RMS for Different Pseudoabsorber Cross Sections (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('RMS')
plt.legend() 
plt.show() 


#%%

#All the RMS (High SZA)

plt.plot(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[5], label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[5], label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[5], label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[5], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[5], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[5], label = 'polynom_order_3_wav_430_470')
plt.plot(ref_224504_redo_analysis[0], ref_224504_redo_analysis[5], label = 'ref_224504')
plt.plot(ref_221505_redo_analysis[0], ref_221505_redo_analysis[5], label = 'ref_221505')
plt.plot(ref_223000_redo_analysis[0], ref_223000_redo_analysis[5], label = 'ref_223000')
plt.plot(ref_215959_redo_analysis[0], ref_215959_redo_analysis[5], label = 'ref_215959')
plt.plot(ref_230002_redo_analysis[0], ref_230002_redo_analysis[5], label = 'ref_230002')
plt.plot(ref_210005_redo_analysis[0], ref_210005_redo_analysis[5], label = 'ref_210005')
plt.plot(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 8')
plt.plot(polynomial_order_7_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 7')
plt.plot(polynomial_order_6_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 6')
plt.plot(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 5')
plt.plot(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[5], label = 'Polynomial Order 4')
plt.plot(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 3')
plt.plot(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 2')
plt.plot(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[5],  label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[5],  label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[5],  label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[5], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[5], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[5],  label = 'polynom_order_3_wav_430_470')
plt.plot(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[5], label = 'NO2_294K_Vandaele')
plt.plot(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[5], label = 'NO2_293K_Burrows')
plt.plot(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[5], label = 'NO2_293K_Bogumil')
plt.plot(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[5], label = 'NO2_273K_Burrows')
plt.plot(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[5], label = 'NO2_273K_Bogumil')
plt.plot(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[5], label = 'NO2_243K_Bogumil')
plt.plot(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[5], label = 'NO2_241K_Burrows')
plt.plot(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[5], label = 'NO2_223K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[5], label = 'NO2_203K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[5], label = 'NO2_220K_Vandaele')
plt.plot(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[5], label = 'O3 ref 193K')
plt.plot(no_shift_redo_analysis[0], no_shift_redo_analysis[5])
plt.plot(no_stretch_redo_analysis[0], no_stretch_redo_analysis[5])
plt.plot(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[5])
plt.plot(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[5])
plt.plot(PCA_poly2_analysis[0], PCA_poly2_analysis[5], label = 'Poly2')
plt.plot(PCA_poly3_analysis[0], PCA_poly3_analysis[5],  label = 'Poly3')
plt.plot(no_PCA_analysis[0], no_PCA_analysis[5], label = 'No PCA')
#plt.plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5], label = 'Best Parameters')
plt.plot(RingSpectrum_analysis[0], RingSpectrum_analysis[5], label = 'Best Parameters')
plt.title('RMS for All Fit Scenarious (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('RMS')
plt.legend(bbox_to_anchor=(1.0, 1.1), ncol = 2) 

plt.savefig('RMS_for_all_fit_scenarios_high_SZA', dpi = 400, bbox_inches='tight')

plt.show() 

#%%


# All the RMS
# Low SZA 

plt.plot(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[5], label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[5], label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[5], label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[5], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[5], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[5], label = 'polynom_order_3_wav_430_470')
plt.plot(ref_224504_redo_low[0], ref_224504_redo_low[5], label = 'ref_224504')
plt.plot(ref_221505_redo_low[0], ref_221505_redo_low[5], label = 'ref_221505')
plt.plot(ref_223000_redo_low[0], ref_223000_redo_low[5], label = 'ref_223000')
plt.plot(ref_215959_redo_low[0], ref_215959_redo_low[5], label = 'ref_215959')
plt.plot(ref_230002_redo_low[0], ref_230002_redo_low[5], label = 'ref_230002')
plt.plot(ref_210005_redo_low[0], ref_210005_redo_low[5], label = 'ref_210005')
plt.plot(polynomial_order_8_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 8')
plt.plot(polynomial_order_7_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 7')
plt.plot(polynomial_order_6_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 6')
plt.plot(polynom_order_5_with_CLD_sigmas_v2_low[0], polynom_order_5_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 5')
plt.plot(polynomial_order_4_redo_low[0], polynomial_order_4_redo_low[5], label = 'Polynomial Order 4')
plt.plot(polynom_order_3_with_CLD_sigmas_v2_low[0], polynom_order_3_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 3')
plt.plot(polynom_order_2_with_CLD_sigmas_v2_low[0], polynom_order_2_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 2')
plt.plot(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[5],  label = 'polynom_order_3_wav_428_468')
plt.plot(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[5],  label = 'polynom_order_3_wav_435_465')
plt.plot(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[5],  label = 'polynom_order_3_wav_425_455')
plt.plot(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[5], label = 'polynom_order_3_wav_410_450')
plt.plot(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[5], label = 'polynom_order_3_wav_420_460')
plt.plot(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[5],  label = 'polynom_order_3_wav_430_470')
plt.plot(NO2_294K_Vandaele_ref_redo_low[0], NO2_294K_Vandaele_ref_redo_low[5], label = 'NO2_294K_Vandaele')
plt.plot(NO2_293K_Burrows_ref_redo_low[0], NO2_293K_Burrows_ref_redo_low[5], label = 'NO2_293K_Burrows')
plt.plot(NO2_293K_Bogumil_ref_redo_low[0], NO2_293K_Bogumil_ref_redo_low[5], label = 'NO2_293K_Bogumil')
plt.plot(NO2_273K_Burrows_ref_redo_low[0], NO2_273K_Burrows_ref_redo_low[5], label = 'NO2_273K_Burrows')
plt.plot(NO2_273K_Bogumil_ref_redo_low[0], NO2_273K_Bogumil_ref_redo_low[5], label = 'NO2_273K_Bogumil')
plt.plot(NO2_243K_Bogumil_ref_redo_low[0], NO2_243K_Bogumil_ref_redo_low[5], label = 'NO2_243K_Bogumil')
plt.plot(NO2_241K_Burrows_ref_redo_low[0], NO2_241K_Burrows_ref_redo_low[5], label = 'NO2_241K_Burrows')
plt.plot(NO2_223K_Bogumil_ref_redo_low[0], NO2_223K_Bogumil_ref_redo_low[5], label = 'NO2_223K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_203K_Bogumi_ref_redo_low[0], NO2_203K_Bogumi_ref_redo_low[5], label = 'NO2_203K_Bogumil')
plt.plot(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[5], label = 'NO2_221K_Burrows')
plt.plot(NO2_220K_Vandaele_redo_low[0], NO2_220K_Vandaele_redo_low[5], label = 'NO2_220K_Vandaele')
plt.plot(O3_193K_ref_redo_low[0], O3_193K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_203K_ref_redo_low[0], O3_203K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_213K_ref_redo_low[0], O3_213K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_223K_ref_redo_low[0], O3_223K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_233K_ref_redo_low[0], O3_233K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_273K_ref_redo_low[0], O3_273K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_253K_ref_redo_low[0], O3_253K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(O3_263K_ref_redo_low[0], O3_263K_ref_redo_low[5], label = 'O3 ref 193K')
plt.plot(no_shift_redo_low[0], no_shift_redo_low[5])
plt.plot(no_stretch_redo_low[0], no_stretch_redo_low[5])
plt.plot(first_order_stretch_redo_low[0], first_order_stretch_redo_low[5])
plt.plot(linear_offset_order_2_redo_low[0], linear_offset_order_2_redo_low[5])
plt.plot(PCA_poly2_low[0], PCA_poly2_low[5], label = 'Poly2')
plt.plot(PCA_poly3_low[0], PCA_poly3_low[5],  label = 'Poly3')
plt.plot(no_PCA_low[0], no_PCA_low[5], label = 'No PCA')
#plt.plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = 'Best Parameters')
plt.plot(RingSpectrum_low[0], RingSpectrum_low[5], label = 'Best Parameters')
plt.title('RMS for All Fit Scenarious (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('RMS')
plt.legend(bbox_to_anchor=(1.0, 1.1), ncol = 2) 

plt.savefig('RMS_for_all_fit_scenarios_low_SZA', dpi = 400, bbox_inches='tight')

plt.show() 




#%%

#Comparing with and without Ring Spectrum (High SZA)
 
plt.title('IO SCDs with and without Ring Spectrum (High SZAs)')
plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9] , label = 'Previous Best')
plt.errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], yerr = RingSpectrum_analysis[9] , label = 'Ring Spectrum')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show()



plt.errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[6], yerr = best_params_with_IO_run2_analysis[7] , label = 'Previous Best')
plt.errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[6], yerr = RingSpectrum_analysis[7] , label = 'Ring Spectrum')
plt.title('CLD with and without Ring Spectrum (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('CLD')
plt.legend() 
plt.show() 

plt.plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5],  label = 'Previous Best')
plt.plot(RingSpectrum_analysis[0], RingSpectrum_analysis[5], label = 'Ring Spectrum')
plt.title('RMS with and without Ring Spectrum (High SZAs)')
plt.xlabel('SZA')
plt.ylabel('RMS')
plt.legend() 
plt.show() 


#%%

#with and without Ring Spectrum (Low SZA)

plt.title('IO SCDs with and without Ring Spectrum (Low SZAs)')
plt.errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9] , label = 'Previous Best')
plt.errorbar(RingSpectrum_low[0], RingSpectrum_low[8], yerr = RingSpectrum_low[9] , label = 'Ring Spectrum')
plt.xlabel('SZA')
plt.ylabel('stable_SlCol_IO')
plt.legend() 
plt.show()



plt.errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[6], yerr = best_params_with_IO_run2_low[7] , label = 'Previous Best')
plt.errorbar(RingSpectrum_low[0], RingSpectrum_low[6], yerr = RingSpectrum_low[7] , label = 'Ring Spectrum')
plt.title('CLD with and without Ring Spectrum (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('CLD')
plt.legend() 
plt.show() 

plt.plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5],  label = 'Previous Best')
plt.plot(RingSpectrum_low[0], RingSpectrum_low[5], label = 'Ring Spectrum')
plt.title('RMS with and without Ring Spectrum (Low SZAs)')
plt.xlabel('SZA')
plt.ylabel('RMS')
plt.legend() 
plt.show() 


#%%

# Creating plots to put in report

# Combined subplots for different variables - low SZA on left, high SZA on right
# Starting with pseudoabsorbers

fig, axs = plt.subplots(2, 2, sharex='col')

plt.suptitle('Pseudoabsorber Cross Sections')


# Low SZAs on left

axs[0, 0].errorbar(PCA_poly2_low[0], PCA_poly2_low[8], yerr =  PCA_poly2_low[9], label = 'Poly2')
axs[0, 0].errorbar(PCA_poly3_low[0], PCA_poly3_low[8], yerr =  PCA_poly3_low[9], label = 'Poly3')
axs[0, 0].errorbar(no_PCA_low[0], no_PCA_low[8], yerr =  no_PCA_low[9], label = 'No PCA')
axs[0, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9] , label = 'Poly1')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')


axs[1,0].plot(PCA_poly2_low[0], PCA_poly2_low[5], label = 'Poly2')
axs[1,0].plot(PCA_poly3_low[0], PCA_poly3_low[5],  label = 'Poly3')
axs[1,0].plot(no_PCA_low[0], no_PCA_low[5], label = 'No PCA')
axs[1,0].plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = 'Poly1')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(PCA_poly2_analysis[0], PCA_poly2_analysis[8], yerr =  PCA_poly2_analysis[9], label = 'Poly2')
axs[0, 1].errorbar(PCA_poly3_analysis[0], PCA_poly3_analysis[8], yerr =  PCA_poly3_analysis[9], label = 'Poly3')
axs[0, 1].errorbar(no_PCA_analysis[0], no_PCA_analysis[8], yerr =  no_PCA_analysis[9], label = 'No PCA')
axs[0, 1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9] , label = 'Poly1')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(-0.2, 0.7, 0.5, 0.5), ncol = 4)


axs[1,1].plot(PCA_poly2_analysis[0], PCA_poly2_analysis[5], label = 'Poly2')
axs[1,1].plot(PCA_poly3_analysis[0], PCA_poly3_analysis[5],  label = 'Poly3')
axs[1,1].plot(no_PCA_analysis[0], no_PCA_analysis[5], label = 'No PCA')
axs[1,1].plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5], label = 'Poly1')
axs[1,1].set_xlabel('SZA (degrees)')

plt.savefig('Pseudoabsorber_Comparison.png', dpi = 400)


plt.show()

#%%

# Combined Plot for All Scenarios

plt.rcParams.update({'font.size': 5, 'errorbar.capsize' : 0, 'lines.markersize' : 0.8, 'lines.marker' : '.', 'lines.linewidth': 0.2})

fig, axs = plt.subplots(5, 2, sharex='col', figsize=(5,8))

#plt.suptitle('Wavelength Ranges')


fig.subplots_adjust(hspace=0.2)

# Low SZAs on left

axs[0, 0].errorbar(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[8], yerr = polynom_order_3_wav_428_468_low[9], label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[0, 0].errorbar(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[8], yerr = polynom_order_3_wav_435_465_low[9], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[0, 0].errorbar(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[8], yerr = polynom_order_3_wav_425_455_low[9], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[0, 0].errorbar(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[8], yerr = polynom_order_3_wav_410_450_redo_low[9], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[0, 0].errorbar(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[8], yerr = polynom_order_3_wav_420_460_redo_low[9], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[0, 0].errorbar(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[8], yerr = polynom_order_3_wav_430_470_redo_low[9], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[0, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9], label = '425 - 465 nm', color = 'pink')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')

axs[1, 0].errorbar(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[1], yerr = polynom_order_3_wav_428_468_low[2], label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[1, 0].errorbar(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[1], yerr = polynom_order_3_wav_435_465_low[2], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[1, 0].errorbar(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[1], yerr = polynom_order_3_wav_425_455_low[2], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[1, 0].errorbar(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[1], yerr = polynom_order_3_wav_410_450_redo_low[2], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[1, 0].errorbar(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[1], yerr = polynom_order_3_wav_420_460_redo_low[2], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[1, 0].errorbar(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[1], yerr = polynom_order_3_wav_430_470_redo_low[2], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[1, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[1], yerr = best_params_with_IO_run2_low[2], label = '425 - 465 nm', color = 'pink')
axs[1, 0].set_ylabel('dSCD O3 \n ($molec/cm^2$)')

axs[2, 0].errorbar(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[3], yerr = polynom_order_3_wav_428_468_low[4], label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[2, 0].errorbar(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[3], yerr = polynom_order_3_wav_435_465_low[4], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[2, 0].errorbar(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[3], yerr = polynom_order_3_wav_425_455_low[4], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[2, 0].errorbar(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[3], yerr = polynom_order_3_wav_410_450_redo_low[4], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[2, 0].errorbar(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[3], yerr = polynom_order_3_wav_420_460_redo_low[4], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[2, 0].errorbar(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[3], yerr = polynom_order_3_wav_430_470_redo_low[4], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[2, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[3], yerr = best_params_with_IO_run2_low[4], label = '425 - 465 nm', color = 'pink')
axs[2, 0].set_ylabel('dSCD NO2 \n ($molec/cm^2$)')

axs[3,0].errorbar(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[6], yerr = polynom_order_3_wav_428_468_low[7], label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[3,0].errorbar(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[6], yerr =  polynom_order_3_wav_435_465_low[7], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[3,0].errorbar(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[6], yerr = polynom_order_3_wav_425_455_low[7], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[3,0].errorbar(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[6], yerr = polynom_order_3_wav_410_450_redo_low[7], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[3,0].errorbar(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[6], yerr = polynom_order_3_wav_420_460_redo_low[7], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[3,0].errorbar(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[6], yerr = polynom_order_3_wav_430_470_redo_low[7], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[3,0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[6], yerr = best_params_with_IO_run2_low[7], label = '425 - 465 nm', color = 'pink')
axs[3,0].set_ylabel('CLD')

axs[4,0].plot(polynom_order_3_wav_428_468_low[0], polynom_order_3_wav_428_468_low[5], label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[4,0].plot(polynom_order_3_wav_435_465_low[0], polynom_order_3_wav_435_465_low[5], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[4,0].plot(polynom_order_3_wav_425_455_low[0], polynom_order_3_wav_425_455_low[5], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[4,0].plot(polynom_order_3_wav_410_450_redo_low[0], polynom_order_3_wav_410_450_redo_low[5], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[4,0].plot(polynom_order_3_wav_420_460_redo_low[0], polynom_order_3_wav_420_460_redo_low[5], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[4,0].plot(polynom_order_3_wav_430_470_redo_low[0], polynom_order_3_wav_430_470_redo_low[5], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[4,0].plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = '425 - 465 nm', color = 'pink')
axs[4,0].set_ylabel('RMS')
axs[4,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[8], yerr = polynom_order_3_wav_428_468_analysis[9], label = '428 - 468', color = 'red', linestyle = 'dotted')
axs[0, 1].errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[8], yerr = polynom_order_3_wav_435_465_analysis[9], label = '435 - 465', color = 'green', linestyle = 'dashed')
axs[0, 1].errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[8], yerr = polynom_order_3_wav_425_455_analysis[9], label = '425 - 455', color = 'blue')
axs[0, 1].errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[8], yerr = polynom_order_3_wav_410_450_redo_analysis[9], label = '410 - 450', color = 'orange')
axs[0, 1].errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[8], yerr = polynom_order_3_wav_420_460_redo_analysis[9], label = '420 - 460', color = 'purple', marker = 'x')
axs[0, 1].errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[8], yerr = polynom_order_3_wav_430_470_redo_analysis[9], label = '430 - 470', color = 'black')
axs[0,1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9], label = '425 - 465', color = 'pink')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(0.4, 1, 0.5, 0.5), ncol = 5)


axs[1, 1].errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[1], yerr = polynom_order_3_wav_428_468_analysis[2], label = '428 - 468 nm', color = 'red', linestyle = 'dotted')
axs[1, 1].errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[1], yerr = polynom_order_3_wav_435_465_analysis[2], label = '435 - 465 nm', color = 'green', linestyle = 'dashed')
axs[1, 1].errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[1], yerr = polynom_order_3_wav_425_455_analysis[2], label = '425 - 455 nm', color = 'blue')
axs[1, 1].errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[1], yerr = polynom_order_3_wav_410_450_redo_analysis[2], label = '410 - 450 nm', color = 'orange')
axs[1, 1].errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[1], yerr = polynom_order_3_wav_420_460_redo_analysis[2], label = '420 - 460 nm', color = 'purple', marker = 'x')
axs[1, 1].errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[1], yerr = polynom_order_3_wav_430_470_redo_analysis[2], label = '430 - 470 nm', color = 'black')
axs[1,1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[1], yerr = best_params_with_IO_run2_analysis[2], label = '425 - 465', color = 'pink')

axs[2, 1].errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[3], yerr = polynom_order_3_wav_428_468_analysis[4], label = '428 - 468', color = 'red', linestyle = 'dotted')
axs[2, 1].errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[3], yerr = polynom_order_3_wav_435_465_analysis[4], label = '435 - 465', color = 'green', linestyle = 'dashed')
axs[2, 1].errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[3], yerr = polynom_order_3_wav_425_455_analysis[4], label = '425 - 455', color = 'blue')
axs[2, 1].errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[3], yerr = polynom_order_3_wav_410_450_redo_analysis[4], label = '410 - 450', color = 'orange')
axs[2, 1].errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[3], yerr = polynom_order_3_wav_420_460_redo_analysis[4], label = '420 - 460', color = 'purple', marker = 'x')
axs[2, 1].errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[3], yerr = polynom_order_3_wav_430_470_redo_analysis[4], label = '430 - 470', color = 'black')
axs[2,1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[3], yerr = best_params_with_IO_run2_analysis[4], label = '425 - 465', color = 'pink')



axs[3,1].errorbar(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[6], yerr = polynom_order_3_wav_428_468_analysis[7],  label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[3,1].errorbar(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[6], yerr = polynom_order_3_wav_435_465_analysis[7], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[3,1].errorbar(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[6], yerr = polynom_order_3_wav_425_455_analysis[7], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[3,1].errorbar(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[6], yerr = polynom_order_3_wav_410_450_redo_analysis[7], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[3,1].errorbar(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[6], yerr = polynom_order_3_wav_420_460_redo_analysis[7], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[3,1].errorbar(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[6], yerr = polynom_order_3_wav_430_470_redo_analysis[7], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[3,1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[6], yerr = best_params_with_IO_run2_analysis[7], label = '425 - 465 nm', color = 'pink')



axs[4,1].plot(polynom_order_3_wav_428_468_analysis[0], polynom_order_3_wav_428_468_analysis[5], label = 'polynom_order_3_wav_428_468', color = 'red', linestyle = 'dotted')
axs[4,1].plot(polynom_order_3_wav_435_465_analysis[0], polynom_order_3_wav_435_465_analysis[5], label = 'polynom_order_3_wav_435_465', color = 'green', linestyle = 'dashed')
axs[4,1].plot(polynom_order_3_wav_425_455_analysis[0], polynom_order_3_wav_425_455_analysis[5], label = 'polynom_order_3_wav_425_455', color = 'blue')
axs[4,1].plot(polynom_order_3_wav_410_450_redo_analysis[0], polynom_order_3_wav_410_450_redo_analysis[5], label = 'polynom_order_3_wav_410_450', color = 'orange')
axs[4,1].plot(polynom_order_3_wav_420_460_redo_analysis[0], polynom_order_3_wav_420_460_redo_analysis[5], label = 'polynom_order_3_wav_420_460', color = 'purple', marker = 'x')
axs[4,1].plot(polynom_order_3_wav_430_470_redo_analysis[0], polynom_order_3_wav_430_470_redo_analysis[5], label = 'polynom_order_3_wav_430_470', color = 'black')
axs[4,1].set_xlabel('SZA (degrees)')
axs[4,1].plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5], label = '425 - 465 nm', color = 'pink')



plt.savefig('Wavelength_Ranges_Comparison.png', dpi = 400,bbox_inches='tight')

plt.show()


#%%

# With and Without Ring Spectrum 

plt.rcParams.update({'font.size': 4, 'errorbar.capsize' : 0, 'lines.markersize' : 0.8, 'lines.marker' : '.', 'lines.linewidth': 0.2})


#plt.suptitle('Wavelength Ranges')


fig.subplots_adjust(hspace=0.2)


fig, axs = plt.subplots(2, 2, sharex='col')

plt.suptitle('With and Without Ring Spectrum')


# Low SZAs on left

axs[0, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9] , label = 'Previous Best', color = 'red')
axs[0, 0].errorbar(RingSpectrum_low[0], RingSpectrum_low[8], yerr = RingSpectrum_low[9] , label = 'Ring Spectrum', color = 'blue')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')


axs[1,0].plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5],  label = 'Previous Best', color = 'red')
axs[1,0].plot(RingSpectrum_low[0], RingSpectrum_low[5], label = 'Ring Spectrum', color = 'blue')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right

axs[0, 1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9] , label = 'Previous Best', color = 'red')
axs[0, 1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], yerr = RingSpectrum_analysis[9] , label = 'Ring Spectrum', color = 'blue')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(-0.3, 0.7, 0.5, 0.5), ncol = 4)

axs[1,1].plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5],  label = 'Previous Best', color = 'red')
axs[1,1].plot(RingSpectrum_analysis[0], RingSpectrum_analysis[5], label = 'Ring Spectrum', color = 'blue')
axs[1,1].set_xlabel('SZA (degrees)')

plt.savefig('RingSpectrum_Comparison.png', dpi = 400, bbox_inches='tight')


plt.show()




#%%

# Best Fit Scenario



fig, axs = plt.subplots(3, 2, sharex='col')

plt.suptitle('Best Fit Scenario')


# Low SZAs on left


axs[0, 0].errorbar(RingSpectrum_low[0], RingSpectrum_low[8], yerr = RingSpectrum_low[9] , label = 'Ring Spectrum')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')


axs[1,0].errorbar(RingSpectrum_low[0], RingSpectrum_low[6], yerr = RingSpectrum_low[7] , label = 'Ring Spectrum')
axs[1,0].set_ylabel('CLD')


axs[2,0].plot(RingSpectrum_low[0], RingSpectrum_low[5], label = 'Ring Spectrum')
axs[2,0].set_ylabel('RMS')
axs[2,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], yerr = RingSpectrum_analysis[9] , label = 'Ring Spectrum')



axs[1,1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[6], yerr = RingSpectrum_analysis[7] , label = 'Ring Spectrum')


axs[2,1].plot(RingSpectrum_analysis[0], RingSpectrum_analysis[5], label = 'Ring Spectrum')
axs[2,1].set_xlabel('SZA (degrees)')

plt.savefig('Best_Fit_Scenario.png', dpi = 400)


plt.show()



#%%

# Reference Spectra

fig, axs = plt.subplots(2, 2, sharex='col')

plt.suptitle('Reference Spectra')




# Low SZAs on left


axs[0, 0].errorbar(ref_230002_redo_low[0], ref_230002_redo_low[8], yerr = ref_230002_redo_low[9], label = 'ref_230002', color = 'pink')
axs[0, 0].errorbar(ref_224504_redo_low[0], ref_224504_redo_low[8], yerr = ref_224504_redo_low[9], label = 'ref_224504', color = 'red')
axs[0, 0].errorbar(ref_223000_redo_low[0], ref_223000_redo_low[8], yerr = ref_223000_redo_low[9], label = 'ref_223000', color = 'blue')
axs[0, 0].errorbar(ref_221505_redo_low[0], ref_221505_redo_low[8], yerr = ref_221505_redo_low[9], label = 'ref_221505', color = 'green')
axs[0, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9], label = '67.75°', color = 'black')
axs[0, 0].errorbar(ref_215959_redo_low[0], ref_215959_redo_low[8], yerr = ref_215959_redo_low[9], label = 'ref_215959', color = 'purple')
axs[0, 0].errorbar(ref_210005_redo_low[0], ref_210005_redo_low[8], yerr = ref_210005_redo_low[9], label = 'ref_210005', color = 'brown')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')

axs[1,0].plot(ref_230002_redo_low[0], ref_230002_redo_low[5], label = 'ref_230002', color = 'pink')
axs[1,0].plot(ref_224504_redo_low[0], ref_224504_redo_low[5], label = 'ref_224504', color = 'red')
axs[1,0].plot(ref_223000_redo_low[0], ref_223000_redo_low[5], label = 'ref_223000', color = 'blue')
axs[1,0].plot(ref_221505_redo_low[0], ref_221505_redo_low[5], label = 'ref_221505', color = 'green')
axs[1,0].plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = '67.75°', color = 'black')
axs[1,0].plot(ref_215959_redo_low[0], ref_215959_redo_low[5], label = 'ref_215959', color = 'purple')
axs[1,0].plot(ref_210005_redo_low[0], ref_210005_redo_low[5], label = 'ref_210005', color = 'brown')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(ref_230002_redo_analysis[0], ref_230002_redo_analysis[8], yerr = ref_230002_redo_analysis[9], label = '77.64°', color = 'pink')
axs[0, 1].errorbar(ref_224504_redo_analysis[0], ref_224504_redo_analysis[8], yerr = ref_224504_redo_analysis[9], label = '75.19°', color = 'red')
axs[0, 1].errorbar(ref_223000_redo_analysis[0], ref_223000_redo_analysis[8], yerr = ref_223000_redo_analysis[9], label = '72.72°', color = 'blue')
axs[0, 1].errorbar(ref_221505_redo_analysis[0], ref_221505_redo_analysis[8], yerr = ref_221505_redo_analysis[9], label = '70.25°', color = 'green')
axs[0, 1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9], label = '67.75°', color = 'black')
axs[0, 1].errorbar(ref_215959_redo_analysis[0], ref_215959_redo_analysis[8], yerr = ref_215959_redo_analysis[9], label = '67.74°', color = 'purple')
axs[0, 1].errorbar(ref_210005_redo_analysis[0], ref_210005_redo_analysis[8], yerr = ref_210005_redo_analysis[9], label = '57.88°', color = 'brown')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(0, 0.7, 0.5, 0.5), ncol = 7)

axs[1,1].plot(ref_230002_redo_analysis[0], ref_230002_redo_analysis[5], label = 'ref_230002', color = 'pink')
axs[1,1].plot(ref_224504_redo_analysis[0], ref_224504_redo_analysis[5], label = 'ref_224504', color = 'red')
axs[1,1].plot(ref_223000_redo_analysis[0], ref_223000_redo_analysis[5], label = 'ref_223000', color = 'blue')
axs[1,1].plot(ref_221505_redo_analysis[0], ref_221505_redo_analysis[5], label = 'ref_221505', color = 'green')
axs[1,1].plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5], label = '67.75°', color = 'black')
axs[1,1].plot(ref_215959_redo_analysis[0], ref_215959_redo_analysis[5], label = 'ref_215959', color = 'purple')
axs[1,1].plot(ref_210005_redo_analysis[0], ref_210005_redo_analysis[5], label = 'ref_210005', color = 'brown')
axs[1,1].set_xlabel('SZA (degrees)')

plt.savefig('Reference_Spectra_Comparison.png', dpi = 400, bbox_inches='tight')

plt.show()



#%%

# Polynomial Orders


fig, axs = plt.subplots(2, 2, sharex='col')

plt.suptitle('Different Order Polynomials')


# Low SZAs on left



axs[0, 0].errorbar(polynomial_order_8_redo_low[0], polynomial_order_8_redo_low[8], yerr = polynomial_order_8_redo_low[9], label = 'Polynomial Order 8', color = 'red')
axs[0, 0].errorbar(polynomial_order_7_redo_low[0], polynomial_order_7_redo_low[8], yerr = polynomial_order_7_redo_low[9], label = 'Polynomial Order 7', color = 'green')
axs[0, 0].errorbar(polynomial_order_6_redo_low[0], polynomial_order_6_redo_low[8], yerr = polynomial_order_6_redo_low[9], label = 'Polynomial Order 6', color = 'blue')
axs[0, 0].errorbar(polynom_order_5_with_CLD_sigmas_v2_low[0], polynom_order_5_with_CLD_sigmas_v2_low[8], yerr = polynom_order_5_with_CLD_sigmas_v2_low[9], label = 'Polynomial Order 5', color = 'purple')
axs[0, 0].errorbar(polynomial_order_4_redo_low[0], polynomial_order_4_redo_low[8], yerr = polynomial_order_4_redo_low[9], label = 'Polynomial Order 4', color = 'pink')
axs[0, 0].errorbar(polynom_order_3_with_CLD_sigmas_v2_low[0], polynom_order_3_with_CLD_sigmas_v2_low[8], yerr = polynom_order_3_with_CLD_sigmas_v2_low[9], label = 'Polynomial Order 3', color = 'brown')
axs[0, 0].errorbar(polynom_order_2_with_CLD_sigmas_v2_low[0], polynom_order_2_with_CLD_sigmas_v2_low[8], yerr = polynom_order_2_with_CLD_sigmas_v2_low[9], label = 'Polynomial Order 2', color = 'black')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')

axs[1,0].plot(polynomial_order_8_redo_low[0], polynomial_order_8_redo_low[5], label = 'Polynomial Order 8', color = 'red')
axs[1,0].plot(polynomial_order_7_redo_low[0], polynomial_order_7_redo_low[5], label = 'Polynomial Order 7', color = 'green')
axs[1,0].plot(polynomial_order_6_redo_low[0], polynomial_order_6_redo_low[5], label = 'Polynomial Order 6', color = 'blue')
axs[1,0].plot(polynom_order_5_with_CLD_sigmas_v2_low[0], polynom_order_5_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 5', color = 'purple')
axs[1,0].plot(polynomial_order_4_redo_low[0], polynomial_order_4_redo_low[5], label = 'Polynomial Order 4', color = 'pink')
axs[1,0].plot(polynom_order_3_with_CLD_sigmas_v2_low[0], polynom_order_3_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 3', color = 'brown')
axs[1,0].plot(polynom_order_2_with_CLD_sigmas_v2_low[0], polynom_order_2_with_CLD_sigmas_v2_low[5], label = 'Polynomial Order 2', color = 'black')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[8], yerr = polynomial_order_8_redo_analysis[9], label = 'Order 8', color = 'red')
axs[0, 1].errorbar(polynomial_order_7_redo_analysis[0], polynomial_order_7_redo_analysis[8], yerr = polynomial_order_7_redo_analysis[9], label = 'Order 7', color = 'green')
axs[0, 1].errorbar(polynomial_order_6_redo_analysis[0], polynomial_order_6_redo_analysis[8], yerr = polynomial_order_6_redo_analysis[9], label = 'Order 6', color = 'blue')
axs[0, 1].errorbar(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[8], yerr = polynom_order_5_with_CLD_sigmas_v2_analysis[9], label = 'Order 5', color = 'purple')
axs[0, 1].errorbar(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[8], yerr = polynomial_order_4_redo_analysis[9], label = 'Order 4', color = 'pink')
axs[0, 1].errorbar(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[8], yerr = polynom_order_3_with_CLD_sigmas_v2_analysis[9], label = 'Order 3', color = 'brown')
axs[0, 1].errorbar(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[8], yerr = polynom_order_2_with_CLD_sigmas_v2_analysis[9], label = 'Order 2', color = 'black')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(0.1, 0.71, 0.5, 0.5), ncol = 8)


axs[1,1].plot(polynomial_order_8_redo_analysis[0], polynomial_order_8_redo_analysis[5], label = 'Polynomial Order 8', color = 'red')
axs[1,1].plot(polynomial_order_7_redo_analysis[0], polynomial_order_7_redo_analysis[5], label = 'Polynomial Order 7', color = 'green')
axs[1,1].plot(polynomial_order_6_redo_analysis[0], polynomial_order_6_redo_analysis[5], label = 'Polynomial Order 6', color = 'blue')
axs[1,1].plot(polynom_order_5_with_CLD_sigmas_v2_analysis[0], polynom_order_5_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 5', color = 'purple')
axs[1,1].plot(polynomial_order_4_redo_analysis[0], polynomial_order_4_redo_analysis[5], label = 'Polynomial Order 4', color = 'pink')
axs[1,1].plot(polynom_order_3_with_CLD_sigmas_v2_analysis[0], polynom_order_3_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 3', color = 'brown')
axs[1,1].plot(polynom_order_2_with_CLD_sigmas_v2_analysis[0], polynom_order_2_with_CLD_sigmas_v2_analysis[5], label = 'Polynomial Order 2', color = 'black')
axs[1,1].set_xlabel('SZA (degrees)')

plt.savefig('Polynomial_Orders.png', dpi = 400, bbox_inches='tight')

plt.show()

#%%

fig, axs = plt.subplots(2, 2, sharex='col')

#plt.suptitle('NO2 Reference Spectra')



axs[0, 0].errorbar(NO2_294K_Vandaele_ref_redo_low[0], NO2_294K_Vandaele_ref_redo_low[8], yerr = NO2_294K_Vandaele_ref_redo_low[9], label = 'NO2_294K_Vandaele', color = 'red')
axs[0, 0].errorbar(NO2_293K_Burrows_ref_redo_low[0], NO2_293K_Burrows_ref_redo_low[8], yerr = NO2_293K_Burrows_ref_redo_low[9], label = 'NO2_293K_Burrows', color = 'green')
axs[0, 0].errorbar(NO2_293K_Bogumil_ref_redo_low[0], NO2_293K_Bogumil_ref_redo_low[8], yerr = NO2_293K_Bogumil_ref_redo_low[9], label = 'NO2_293K_Bogumil', color = 'blue')
axs[0, 0].errorbar(NO2_273K_Burrows_ref_redo_low[0], NO2_273K_Burrows_ref_redo_low[8], yerr = NO2_273K_Burrows_ref_redo_low[9], label = 'NO2_273K_Burrows', color = 'purple')
axs[0, 0].errorbar(NO2_273K_Bogumil_ref_redo_low[0], NO2_273K_Bogumil_ref_redo_low[8], yerr = NO2_273K_Bogumil_ref_redo_low[9], label = 'NO2_273K_Bogumil', color = 'pink')
axs[0, 0].errorbar(NO2_243K_Bogumil_ref_redo_low[0], NO2_243K_Bogumil_ref_redo_low[8], yerr = NO2_243K_Bogumil_ref_redo_low[9], label = 'NO2_243K_Bogumil', color = 'brown')
axs[0, 0].errorbar(NO2_241K_Burrows_ref_redo_low[0], NO2_241K_Burrows_ref_redo_low[8], yerr = NO2_241K_Burrows_ref_redo_low[9], label = 'NO2_241K_Burrows', color = 'black')
axs[0, 0].errorbar(NO2_223K_Bogumil_ref_redo_low[0], NO2_223K_Bogumil_ref_redo_low[8], yerr = NO2_223K_Bogumil_ref_redo_low[9], label = 'NO2_223K_Bogumil', color = 'cyan')
axs[0, 0].errorbar(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[8], yerr = NO2_221K_Burrows_ref_redo_low[9], label = 'NO2_221K_Burrows', color = 'orange')
axs[0, 0].errorbar(NO2_203K_Bogumi_ref_redo_low[0], NO2_203K_Bogumi_ref_redo_low[8], yerr = NO2_203K_Bogumi_ref_redo_low[9], label = 'NO2_203K_Bogumil', color = 'olive')
axs[0, 0].errorbar(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[8], yerr = NO2_221K_Burrows_ref_redo_low[9], label = 'NO2_221K_Burrows', color = 'grey')
axs[0, 0].errorbar(NO2_220K_Vandaele_redo_low[0], NO2_220K_Vandaele_redo_low[8], yerr = NO2_220K_Vandaele_redo_low[9], label = 'NO2_220K_Vandaele')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')


axs[1,0].plot(NO2_294K_Vandaele_ref_redo_low[0], NO2_294K_Vandaele_ref_redo_low[5], label = 'NO2_294K_Vandaele', color = 'red')
axs[1,0].plot(NO2_293K_Burrows_ref_redo_low[0], NO2_293K_Burrows_ref_redo_low[5], label = 'NO2_293K_Burrows', color = 'green')
axs[1,0].plot(NO2_293K_Bogumil_ref_redo_low[0], NO2_293K_Bogumil_ref_redo_low[5], label = 'NO2_293K_Bogumil', color = 'blue')
axs[1,0].plot(NO2_273K_Burrows_ref_redo_low[0], NO2_273K_Burrows_ref_redo_low[5], label = 'NO2_273K_Burrows', color = 'purple')
axs[1,0].plot(NO2_273K_Bogumil_ref_redo_low[0], NO2_273K_Bogumil_ref_redo_low[5], label = 'NO2_273K_Bogumil', color = 'pink')
axs[1,0].plot(NO2_243K_Bogumil_ref_redo_low[0], NO2_243K_Bogumil_ref_redo_low[5], label = 'NO2_243K_Bogumil', color = 'brown')
axs[1,0].plot(NO2_241K_Burrows_ref_redo_low[0], NO2_241K_Burrows_ref_redo_low[5], label = 'NO2_241K_Burrows', color = 'black')
axs[1,0].plot(NO2_223K_Bogumil_ref_redo_low[0], NO2_223K_Bogumil_ref_redo_low[5], label = 'NO2_223K_Bogumil', color = 'cyan')
axs[1,0].plot(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[5], label = 'NO2_221K_Burrows', color = 'orange')
axs[1,0].plot(NO2_203K_Bogumi_ref_redo_low[0], NO2_203K_Bogumi_ref_redo_low[5], label = 'NO2_203K_Bogumil', color = 'olive')
axs[1,0].plot(NO2_221K_Burrows_ref_redo_low[0], NO2_221K_Burrows_ref_redo_low[5], label = 'NO2_221K_Burrows', color = 'grey')
axs[1,0].plot(NO2_220K_Vandaele_redo_low[0], NO2_220K_Vandaele_redo_low[5], label = 'NO2_220K_Vandaele', color = 'teal')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[8], yerr = NO2_294K_Vandaele_ref_redo_analysis[9], label = '294K Vandaele', color = 'red')
axs[0, 1].errorbar(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[8], yerr = NO2_293K_Burrows_ref_redo_analysis[9], label = '293K Burrows', color = 'green')
axs[0, 1].errorbar(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[8], yerr = NO2_293K_Bogumil_ref_redo_analysis[9], label = '293K Bogumil', color = 'blue')
axs[0, 1].errorbar(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[8], yerr = NO2_273K_Burrows_ref_redo_analysis[9], label = '273K Burrows', color = 'purple')
axs[0, 1].errorbar(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[8], yerr = NO2_273K_Bogumil_ref_redo_analysis[9], label = '273K Bogumil', color = 'pink')
axs[0, 1].errorbar(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[8], yerr = NO2_243K_Bogumil_ref_redo_analysis[9], label = '243K Bogumil', color = 'brown')
axs[0, 1].errorbar(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[8], yerr = NO2_241K_Burrows_ref_redo_analysis[9], label = '241K Burrows', color = 'black')
axs[0, 1].errorbar(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[8], yerr = NO2_223K_Bogumil_ref_redo_analysis[9], label = '223K Bogumil', color = 'cyan')
axs[0, 1].errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8], yerr = NO2_221K_Burrows_ref_redo_analysis[9], label = '221K Burrows', color = 'orange')
axs[0, 1].errorbar(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[8], yerr = NO2_203K_Bogumi_ref_redo_analysis[9], label = '203K Bogumil', color = 'olive')
axs[0, 1].errorbar(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[8], yerr = NO2_221K_Burrows_ref_redo_analysis[9], label = '221K Burrows', color = 'grey')
axs[0, 1].errorbar(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[8], yerr = NO2_220K_Vandaele_redo_analysis[9], label = '220K Vandaele', color = 'teal')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(0.4, 0.8, 0.5, 0.5), ncol = 6)





axs[1,1].plot(NO2_294K_Vandaele_ref_redo_analysis[0], NO2_294K_Vandaele_ref_redo_analysis[5], label = 'NO2_294K_Vandaele', color = 'red')
axs[1,1].plot(NO2_293K_Burrows_ref_redo_analysis[0], NO2_293K_Burrows_ref_redo_analysis[5], label = 'NO2_293K_Burrows', color = 'green')
axs[1,1].plot(NO2_293K_Bogumil_ref_redo_analysis[0], NO2_293K_Bogumil_ref_redo_analysis[5], label = 'NO2_293K_Bogumil', color = 'blue')
axs[1,1].plot(NO2_273K_Burrows_ref_redo_analysis[0], NO2_273K_Burrows_ref_redo_analysis[5], label = 'NO2_273K_Burrows', color = 'purple')
axs[1,1].plot(NO2_273K_Bogumil_ref_redo_analysis[0], NO2_273K_Bogumil_ref_redo_analysis[5], label = 'NO2_273K_Bogumil', color = 'pink')
axs[1,1].plot(NO2_243K_Bogumil_ref_redo_analysis[0], NO2_243K_Bogumil_ref_redo_analysis[5], label = 'NO2_243K_Bogumil', color = 'brown')
axs[1,1].plot(NO2_241K_Burrows_ref_redo_analysis[0], NO2_241K_Burrows_ref_redo_analysis[5], label = 'NO2_241K_Burrows', color = 'black')
axs[1,1].plot(NO2_223K_Bogumil_ref_redo_analysis[0], NO2_223K_Bogumil_ref_redo_analysis[5], label = 'NO2_223K_Bogumil', color = 'cyan')
axs[1,1].plot(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[5], label = 'NO2_221K_Burrows', color = 'orange')
axs[1,1].plot(NO2_203K_Bogumi_ref_redo_analysis[0], NO2_203K_Bogumi_ref_redo_analysis[5], label = 'NO2_203K_Bogumil', color = 'olive')
axs[1,1].plot(NO2_221K_Burrows_ref_redo_analysis[0], NO2_221K_Burrows_ref_redo_analysis[5], label = 'NO2_221K_Burrows', color = 'grey')
axs[1,1].plot(NO2_220K_Vandaele_redo_analysis[0], NO2_220K_Vandaele_redo_analysis[5], label = 'NO2_220K_Vandaele', color = 'teal')
axs[1,1].set_xlabel('SZA (degrees)')

plt.savefig('NO2_Reference_Spectra.png', dpi = 400, bbox_inches='tight')

plt.show()


    #%%



fig, axs = plt.subplots(2, 2, sharex='col')

plt.suptitle('O3 Reference Spectra')

#Low Reference Spectra on Left

#O3 reference spectra low SZAS

axs[0, 0].errorbar(O3_193K_ref_redo_low[0], O3_193K_ref_redo_low[8], yerr = O3_193K_ref_redo_low[9], label = 'O3 ref 193K', color = 'red')
axs[0, 0].errorbar(O3_203K_ref_redo_low[0], O3_203K_ref_redo_low[8], yerr = O3_203K_ref_redo_low[9], label = 'O3 ref 203K', color = 'green')
axs[0, 0].errorbar(O3_213K_ref_redo_low[0], O3_213K_ref_redo_low[8], yerr = O3_213K_ref_redo_low[9], label = 'O3 ref 213K', color = 'blue')
axs[0, 0].errorbar(O3_223K_ref_redo_low[0], O3_223K_ref_redo_low[8], yerr = O3_223K_ref_redo_low[9], label = 'O3 ref 223K', color = 'purple')
axs[0, 0].errorbar(O3_233K_ref_redo_low[0], O3_233K_ref_redo_low[8], yerr = O3_233K_ref_redo_low[9], label = 'O3 ref 233K', color = 'pink')
axs[0, 0].errorbar(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[8], yerr = best_params_with_IO_run2_low[9], label = '243K', color = 'brown')
axs[0, 0].errorbar(O3_253K_ref_redo_low[0], O3_253K_ref_redo_low[8], yerr = O3_253K_ref_redo_low[9], label = 'O3 ref 253K', color = 'black')
axs[0, 0].errorbar(O3_263K_ref_redo_low[0], O3_263K_ref_redo_low[8], yerr = O3_263K_ref_redo_low[9], label = 'O3 ref 263K', color = 'cyan')
axs[0, 0].errorbar(O3_273K_ref_redo_low[0], O3_273K_ref_redo_low[8], yerr = O3_273K_ref_redo_low[9], label = 'O3 ref 273K', color = 'orange')
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')



axs[1,0].plot(O3_193K_ref_redo_low[0], O3_193K_ref_redo_low[5], label = 'O3 ref 193K', color = 'red')
axs[1,0].plot(O3_203K_ref_redo_low[0], O3_203K_ref_redo_low[5], label = 'O3 ref 203K', color = 'green')
axs[1,0].plot(O3_213K_ref_redo_low[0], O3_213K_ref_redo_low[5], label = 'O3 ref 213K', color = 'blue')
axs[1,0].plot(O3_223K_ref_redo_low[0], O3_223K_ref_redo_low[5], label = 'O3 ref 223K', color = 'purple')
axs[1,0].plot(O3_233K_ref_redo_low[0], O3_233K_ref_redo_low[5], label = 'O3 ref 233K', color = 'pink')
axs[0,0].plot(best_params_with_IO_run2_low[0], best_params_with_IO_run2_low[5], label = '243K', color = 'brown')
axs[1,0].plot(O3_253K_ref_redo_low[0], O3_253K_ref_redo_low[5], label = 'O3 ref 253K', color = 'black')
axs[1,0].plot(O3_263K_ref_redo_low[0], O3_263K_ref_redo_low[5], label = 'O3 ref 263K', color = 'cyan')
axs[1,0].plot(O3_273K_ref_redo_low[0], O3_273K_ref_redo_low[5], label = 'O3 ref 273K', color = 'orange')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right

#O3 reference spectra high SZAs


axs[0, 1].errorbar(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[8], yerr = O3_193K_ref_redo_analysis[9], label = '193K', color = 'red')
axs[0, 1].errorbar(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[8], yerr = O3_203K_ref_redo_analysis[9], label = '203K', color = 'green')
axs[0, 1].errorbar(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[8], yerr = O3_213K_ref_redo_analysis[9], label = '213K', color = 'blue')
axs[0, 1].errorbar(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[8], yerr = O3_223K_ref_redo_analysis[9], label = '223K', color = 'purple')
axs[0, 1].errorbar(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[8], yerr = O3_233K_ref_redo_analysis[9], label = '233K', color = 'pink')
axs[0, 1].errorbar(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[8], yerr = best_params_with_IO_run2_analysis[9], label = '243K', color = 'brown')

axs[0, 1].errorbar(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[8], yerr = O3_253K_ref_redo_analysis[9], label = '253K', color = 'black')
axs[0, 1].errorbar(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[8], yerr = O3_263K_ref_redo_analysis[9], label = '263K', color = 'cyan')
axs[0, 1].errorbar(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[8], yerr = O3_273K_ref_redo_analysis[9], label = '273K', color = 'orange')
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(0.1, 0.7, 0.5, 0.5), ncol = 9)




axs[1,1].plot(O3_193K_ref_redo_analysis[0], O3_193K_ref_redo_analysis[5], label = 'O3 ref 193K', color = 'red')
axs[1,1].plot(O3_203K_ref_redo_analysis[0], O3_203K_ref_redo_analysis[5], label = 'O3 ref 203K', color = 'green')
axs[1,1].plot(O3_213K_ref_redo_analysis[0], O3_213K_ref_redo_analysis[5], label = 'O3 ref 213K', color = 'blue')
axs[1,1].plot(O3_223K_ref_redo_analysis[0], O3_223K_ref_redo_analysis[5], label = 'O3 ref 223K', color = 'purple')
axs[1,1].plot(O3_233K_ref_redo_analysis[0], O3_233K_ref_redo_analysis[5], label = 'O3 ref 233K', color = 'pink')
axs[1,1].plot(best_params_with_IO_run2_analysis[0], best_params_with_IO_run2_analysis[5], label = '243K', color = 'brown')
axs[1,1].plot(O3_253K_ref_redo_analysis[0], O3_253K_ref_redo_analysis[5], label = 'O3 ref 253K', color = 'black')
axs[1,1].plot(O3_263K_ref_redo_analysis[0], O3_263K_ref_redo_analysis[5], label = 'O3 ref 263K', color = 'cyan')
axs[1,1].set_xlabel('SZA (degrees)')
axs[1,1].plot(O3_273K_ref_redo_analysis[0], O3_273K_ref_redo_analysis[5], label = 'O3 ref 273K', color = 'orange')

plt.savefig('O3_Reference_Spectra.png', dpi = 400, bbox_inches='tight')

plt.show()


#%%



fig, axs = plt.subplots(2, 2, sharex='col')

plt.suptitle('Shift and Stretch')

#Low Reference Spectra on Left

axs[0, 0].errorbar(no_shift_redo_low[0], no_shift_redo_low[8], yerr =  no_shift_redo_low[9], label = 'No shift', color = 'red', linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 0].errorbar(no_stretch_redo_low[0], no_stretch_redo_low[8], yerr = no_stretch_redo_low[9], label = 'No Stretch', color = 'green',  linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 0].errorbar(first_order_stretch_redo_low[0], first_order_stretch_redo_low[8], yerr = first_order_stretch_redo_low[9], label = 'First Order Stretch', color = 'blue',  linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 0].errorbar(linear_offset_order_2_redo_low[0], linear_offset_order_2_redo_low[8], yerr = linear_offset_order_2_redo_low[9], label = '2nd Order Linear Offset', color = 'purple',  linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 0].errorbar(RingSpectrum_low[0], RingSpectrum_low[8], yerr = RingSpectrum_low[9] , label = 'Best Parameters', color = 'orange',  linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')




axs[1,0].plot(no_shift_redo_low[0], no_shift_redo_low[5], label = 'No shift', color = 'red', linewidth = 0, marker = 'x')
axs[1,0].plot(no_stretch_redo_low[0], no_stretch_redo_low[5], label = 'No Stretch', color = 'green', linewidth = 0, marker = 'x')
axs[1,0].plot(first_order_stretch_redo_low[0], first_order_stretch_redo_low[5], label = 'First Order Stretch', color = 'blue', linewidth = 0, marker = 'x')
axs[1,0].plot(linear_offset_order_2_redo_low[0], linear_offset_order_2_redo_low[5], label = '2nd Order Linear Offset', color = 'purple', linewidth = 0, marker = 'x')
axs[1,0].plot(RingSpectrum_low[0], RingSpectrum_low[5], label = 'Best Parameters',  color = 'orange', linewidth = 0, marker = 'x')
axs[1,0].set_ylabel('RMS')
axs[1,0].set_xlabel('SZA (degrees)')


# High SZAs on Right


axs[0, 1].errorbar(no_shift_redo_analysis[0], no_shift_redo_analysis[8], yerr =  no_shift_redo_analysis[9], label = 'No Shift', color = 'red', linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 1].errorbar(no_stretch_redo_analysis[0], no_stretch_redo_analysis[8], yerr = no_stretch_redo_analysis[9], label = 'No Stretch', color = 'green', linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 1].errorbar(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[8], yerr = first_order_stretch_redo_analysis[9], label = 'Stretch Order 1', color = 'blue', linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 1].errorbar(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[8], yerr = linear_offset_order_2_redo_analysis[9], label = 'Linear Offset Order 2', color = 'purple', linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], yerr = RingSpectrum_analysis[9] , label = 'Stretch Order 2, No shift',  color = 'orange', linewidth = 0, marker = 'x', elinewidth = 0.3, capsize = 0.1)
axs[0, 1].legend(loc='upper right',bbox_to_anchor=(0.2, 0.72, 0.5, 0.5), ncol = 5)



axs[1,1].plot(no_shift_redo_analysis[0], no_shift_redo_analysis[5], color = 'red', linewidth = 0, marker = 'x')
axs[1,1].plot(no_stretch_redo_analysis[0], no_stretch_redo_analysis[5], color = 'green', linewidth = 0, marker = 'x')
axs[1,1].plot(first_order_stretch_redo_analysis[0], first_order_stretch_redo_analysis[5], color = 'blue', linewidth = 0, marker = 'x')
axs[1,1].plot(linear_offset_order_2_redo_analysis[0], linear_offset_order_2_redo_analysis[5], color = 'purple', linewidth = 0, marker = 'x')
axs[1,1].plot(RingSpectrum_analysis[0], RingSpectrum_analysis[5], color = 'orange', linewidth = 0, marker = 'x')
axs[1,1].set_xlabel('SZA (degrees)')

plt.savefig('Shift_and_Stretch.png', dpi = 400, bbox_inches='tight')

plt.show()


#%%


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
plt.ylabel('dSCD IO ($molec/cm^2$)')
plt.grid()
#plt.legend(bbox_to_anchor=(1.0, 1.1), ncol = 2) 

plt.savefig('IO_col_high_SZA', dpi = 400, bbox_inches='tight')

plt.show() 


#%%



# Full plot for best scenario

plt.rcParams.update({'font.size': 5, 'errorbar.capsize' : 0.1, 'lines.markersize' : 0.8, 'lines.marker' : '.', 'lines.linewidth': 0.3})
                

fig, axs = plt.subplots(6, 2, sharex='col', figsize=(5,7))


fig.subplots_adjust(hspace=0.2)


#fig.tight_layout(pad=1.0)

#plt.suptitle('Best Fit Scenario')


# Low SZAs on left


axs[0, 0].errorbar(RingSpectrum_low[0], RingSpectrum_low[8], yerr = RingSpectrum_low[9] , label = 'Ring Spectrum', elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
axs[0, 0].set_ylabel('dSCD IO \n ($molec/cm^2$)')


axs[1,0].errorbar(RingSpectrum_low[0], RingSpectrum_low[6], yerr = RingSpectrum_low[7] , label = 'Ring Spectrum',  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
axs[1,0].set_ylabel('CLD')

axs[2,0].plot(RingSpectrum_low[0], RingSpectrum_low[5], label = 'Ring Spectrum',  marker = 'x',  color = 'blue', linewidth = 0)
axs[2,0].set_ylabel('RMS')


axs[3,0].errorbar(RingSpectrum_low[0], RingSpectrum_low[1],yerr = RingSpectrum_low[2],  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
axs[3,0].set_ylabel('dSCD $O_3$ \n ($molec/cm^2$)')

axs[4,0].errorbar(RingSpectrum_low[0], RingSpectrum_low[3],yerr = RingSpectrum_low[4],  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
axs[4,0].set_ylabel('dSCD $NO_2 \n ($molec/cm^2$)')

axs[5,0].errorbar(RingSpectrum_low[0], RingSpectrum_low[10],yerr = RingSpectrum_low[11],  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
axs[5,0].set_ylabel('dSCD $O_4$ \n ($molec/cm^2$)')
axs[5,0].set_xlabel('SZA (degrees)')

# High SZAs on Right


axs[0, 1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[8], yerr = RingSpectrum_analysis[9] , label = 'Ring Spectrum',  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)



axs[1,1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[6], yerr = RingSpectrum_analysis[7] , label = 'Ring Spectrum',  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)


axs[2,1].plot(RingSpectrum_analysis[0], RingSpectrum_analysis[5], label = 'Ring Spectrum',   marker = 'x',  color = 'blue', linewidth = 0)


axs[3,1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[1],yerr = RingSpectrum_analysis[2],  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
#axs[3,1].set_ylabel('O3')

axs[4,1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[3],yerr = RingSpectrum_analysis[4],  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)

axs[5,1].errorbar(RingSpectrum_analysis[0], RingSpectrum_analysis[10],yerr = RingSpectrum_analysis[11],  elinewidth = 0.3, marker = 'x', ecolor = 'purple', color = 'blue', linewidth = 0)
axs[5,1].set_xlabel('SZA (degrees)')

plt.savefig('Best_Fit_Scenario_Full_Plot.png', dpi = 500, bbox_inches='tight')


plt.show()














    