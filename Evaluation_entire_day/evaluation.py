# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
#from pandas.compat import StringIO


#%%

#File, Datum, StartZeit, SZA, AMF, ChiSquare, \
#Fraunhofer_reference_log, Fraunhofer_reference_log_Error, \
#Ring_Spectrum, Ring_Spectrum_Error, convolution_result_H2O, \
#convolution_result_H2O_Error, convolution_result_NO2, convolution_result_NO2_Error, \
#convolution_result_O3, convolution_result_O3_Error, \
#convolution_result_O4, convolution_result_O4_Error, twat \
#= np.loadtxt('ozonres.dat', skiprows = 1, unpack = True, delimiter = '\t', dtype = str)


#%%

o3_df = pd.read_csv('ozonres.dat', sep='\t')
no2_df = pd.read_csv('no2res.dat', sep='\t')

#%%


#o3_df.plot(x = 'AMF  ', y = 'convolution_result_O3')

AMF_array = o3_df['AMF  ']
conv_res_O3_array = o3_df['convolution_result_O3']
conv_res_O3_array_error = o3_df['convolution_result_O3 Error ']
o3_df['timestamp'] = o3_df['Datum ']+o3_df['StartZeit ']
o3_df['timestamp'] = o3_df['timestamp'].apply(pd.to_datetime, format='%d.%m.%Y%H:%M:%S')
time_array = o3_df['timestamp']

#%%

amf_conc_o3 = np.stack((AMF_array, conv_res_O3_array), axis=-1)

print(len(amf_conc_o3))

amf_O3_below_5 = []
conc_O3_below_5 = []
error_O3_below_5 = []
time_array_below_5 = []
for i in range(len(amf_conc_o3[:,0])):
    if amf_conc_o3[:,0][i] < 5:
        amf_O3_below_5.append(AMF_array[i])
        conc_O3_below_5.append(conv_res_O3_array[i])
        time_array_below_5.append(time_array[i])
        error_O3_below_5.append(conv_res_O3_array_error[i])

print(len(amf_O3_below_5))
plt.figure()
plt.plot(amf_O3_below_5, conc_O3_below_5,'*')
plt.show()

#amf_O3_below_5_above_1_5 = []
#conc_O3_below_5_above_1_5 = []
#time_array_below_5_above_1_5 = []
#for i in range(len(amf_conc_o3[:,0])):
 #   if amf_conc_o3[:,0][i] < 5 and amf_conc_o3[:,0][i] > 1.5:
  #      amf_O3_below_5_above_1_5.append(AMF_array[i])
   #     conc_O3_below_5_above_1_5.append(conv_res_O3_array[i])
    #    time_array_below_5_above_1_5.append(time_array[i])

#print(len(amf_O3_below_5))
plt.figure()
plt.plot(amf_O3_below_5, conc_O3_below_5,'*')
plt.show()


#print(len(amf_O3_below_5_above_1_5))
#plt.figure()
#plt.plot(amf_O3_below_5_above_1_5, conc_O3_below_5_above_1_5,'*')
#plt.show()

#%%

# objective function
def objective(x, a, b):
 return a * x + b


...
# fit curve
popt, pcov = curve_fit(objective, amf_O3_below_5, conc_O3_below_5, p0 = [2*10**18, 10**19])



...
# define new input values
x_new = np.linspace(0,5,1000)
# unpack optima parameters for the objective function
a, b = popt
# use optimal parameters to calculate new values
y_new = objective(x_new, a, b)
print(a, b)
print(pcov)


plt.plot(x_new, y_new, label='Fit to 5 > AMF')
plt.plot(amf_O3_below_5, conc_O3_below_5,'*', label='Data')
plt.legend()
plt.ylabel('Concentration of O$_3$ / molecules / cm$^2$')
plt.xlabel('AMF')
plt.title('Langley plot')
plt.savefig('Langley.png')


#%%

S_F = b * np.ones(len(conc_O3_below_5))
SCD = conc_O3_below_5 - S_F
VCD = SCD/amf_O3_below_5
Delta_SCD_ref = np.sqrt(pcov[1,1]) * np.ones(len(conc_O3_below_5))
Delta_SCD = np.array(error_O3_below_5) 
Delta_VCD =  np.sqrt(Delta_SCD_ref**2 + Delta_SCD**2) / amf_O3_below_5 


plt.errorbar(time_array_below_5, VCD, Delta_VCD)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel('VCD')
plt.xlabel('Time')
plt.title('VCD')
plt.savefig('VCD.png')
plt.show()

VCD_DU = VCD / (2.6*10**16)
VCD_err_DU = Delta_VCD / (2.6*10**16)

plt.errorbar(time_array_below_5, VCD_DU, VCD_err_DU)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.ylabel('VCD / DU')
plt.xlabel('Time')
plt.title('VCD')
plt.savefig('VCD_DU.png')




























