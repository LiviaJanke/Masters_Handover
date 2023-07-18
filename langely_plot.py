# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 13:33:58 2023

@author: lme19
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
## Add user-defined ../lib to system path
#lib_path = os.path.abspath(os.path.join('../lib'))
#if lib_path not in sys.path:
#    sys.path.append(lib_path)
   
# Add user-defined ../data to system path
#data_path = os.path.abspath(os.path.join('../data'))
data_path = os.path.abspath(os.path.join(r'C:\Users\lme19\Documents\Masters\Validation'))
if data_path not in sys.path:
    sys.path.append(data_path)
    
# Global plot parameters    
fheight = 7  # figure height
fwidth = 15  # figure width
fs = 14     # fontsize

# Epsilon definition
eps=1E-10
# Infinity definition
infty=1E99

#%%


amf = np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km_full.amf') 


# SCDs, uncertainties and ancillary data (m)
date, tmp, sza, balloon, scd, scderr = np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km_full.scd',skiprows=2, unpack=True, delimiter = ' ') 
# Translate time string into decimal hour [UTC]


df_scd = pd.read_csv(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km_full.scd', skiprows = 2, delimiter = ' ', dtype = str,  names = ['Date', 'Time/UTC', 'SZA/DEG', 'BALLOON_HEIGHT/KM', 'SCD/MOLEC_CM2', 'SCDERR/MOLEC_CM2'])

#date_and_time = df_scd['Date'] + ' ' + df_scd['Time/UTC']
datetime = pd.to_datetime(df_scd['Date'] + ' ' + df_scd['Time/UTC'])


#vcd, vcd_err = np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km.vcd',skiprows=2, unpack =True) 

# The above VCD stuff isn't right


# should be using below file somehow

Teilchendichte_cm3, Fehler, mitt_luftdichte_cm3, MischVerh, Fehler, von_km, bis_km =  np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km_full.vd',skiprows=2, unpack =True) 


# Layer height is the height of the layer itself
# Not total height

# uppper limit - lower limit

layer_heights = bis_km - von_km

#%%


seg = np.genfromtxt(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km_full.seg', delimiter = ' ') 
# Generate mid-point layers (n)
height = np.array([seg[i]/2.+seg[i+1]/2 for i in range(len(seg)-1)]) # Height layers (mid-of-levels)
#all_heights = np.append(height, 80)

# Dimensions
#n = len(all_heights) # parameter vector dimension
n = len(height)
m = len(scd)    # measurement vector dimension

# Define the inverse problem notation
K = amf # forward model matrix
#K = amf_clean
y = scd # measurement vector
yerr = scderr # measurment uncertainties


#%%

# Plot some rows of the AMF matrix
fig = plt.figure(figsize=(fheight,fwidth))
for i in range(0,m,250):
    plt.plot(amf[i,:],height,label='AMF %d'%(i))
plt.ylabel('Height / k', fontsize=fs)
plt.xlabel('Air mass factor', fontsize=fs)
plt.legend(fontsize=fs);


#%%

# Plot the SCDs and ancillary information
fig, ax = plt.subplots(figsize=(fheight,fwidth))
ax.errorbar(scd,datetime,xerr=scderr,color='k',marker='.')
ax.set_xlabel('SCD / (molec / cm$^2$)', fontsize=fs)
ax.set_ylabel('Time UTC / h', fontsize=fs)
ax.invert_yaxis()
ax2=ax.twiny()
ax2.plot(sza,datetime,'r-')
ax2.set_xlabel('SZA / $^\circ$', fontsize=fs);

#%%

# Langley plot
# Take SZAs up to 86
# Start from 82 or so?

# Air mass SCD_air  = sum over AMF * VD


layer_densities = mitt_luftdichte_cm3 * layer_heights * 1e5

scd_air = np.matmul(amf, layer_densities)

plt.plot(scd_air, scd, linewidth = 0, marker = 'x')

# Better

#%%

plt.plot(scd_air, scd)

#%%

plt.plot(sza, scd_air)

#%%

#fit to plot to VMR (slope) and offset SCD_ref

# Try with SZAs 80 to 88

# Find the closest index to these values computationally

# index 2623 = sza 80

# index 2746 = SZA 82

# index 3112 = SZA 88

# index 2991 = SZA 86

# Try from 84 - points that lie above 0

# 

# Try to index 2991 maybe?

# Karolin used 82-88 
# and 86 - 89

#%%


plt.plot(scd_air[2623:3112], sza[2623:3112], linewidth = 0, marker = 'x')


#%%

# Rudimentary straight line fit

x_testarray = np.linspace(scd_air[1000], scd_air[3200], 1000)

xvals = scd_air[2623:3112]

yvals_err = scderr[2623:3112]

yvals = scd[2623:3112]

p, cov = np.polyfit(xvals,yvals,1, cov = True,  w = 1/yvals_err)
m = p[0]
b = p[1]

plt.plot(xvals, yvals, 'yo', x_testarray, m*x_testarray+b, '--k')
plt.show()

scd_ref = -1 * b

vmr = m

uncerts = (np.sqrt(np.diag(cov)))

print('offset is')
print(scd_ref)
print(uncerts[1])
print('VMR is')
print(vmr)
print(uncerts[0])

#%%

# linear regression fit

X = xvals.reshape(-1, 1)
Y = yvals.reshape(-1, 1)

reg = LinearRegression().fit(X, Y, sample_weight=(yvals_err))


score = reg.score(X, Y, sample_weight=(yvals_err))

coeffs = reg.coef_

intercept = reg.intercept_ * -1


x_testarray = (np.linspace(0, 1.4e25, 100000)).reshape(-1, 1)

y_predicted = reg.predict(x_testarray)

plt.plot(x_testarray, y_predicted, label = 'Linear Regression')
plt.errorbar(scd_air[1000:-80], scd[1000:-80], yerr = scderr[1000:-80], label = 'O3 dSCD', capsize = 1, linewidth = 0.1, marker = '.')
plt.errorbar(scd_air[2623:3112], scd[2623:3112], yerr = scderr[2623:3112], label = 'O3 dSCD for Regression', capsize = 1, linewidth = 0.1, marker = '.')

plt.ylabel('dSCD in Molecules/cm2')
plt.xlabel('Air Mass in Molecules/cm2')
plt.legend()
plt.title('Langley Plot for Ozone')
plt.grid()
plt.savefig('Langley_plot_ozone', dpi = 400)
plt.show()


print(coeffs)
print(intercept)
print(score)

#%%

























