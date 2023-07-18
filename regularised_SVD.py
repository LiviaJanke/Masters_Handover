# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:26:07 2023

@author: lme19
"""

# From Tutorial Paper 5 of Inverse Methods course

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd



import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from datetime import datetime, timezone, timedelta
import glob
from scipy.interpolate import interp1d
#from pysolar.solar import get_altitude

#from pytz import reference
from pysolar.solar import get_altitude

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

# AMF matrix (mxn)
#amf = np.genfromtxt('sunset.amf',skip_header=1) 

#amf = np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\SZA_90_95.amf') 
#amf_clean = np.delete(amf, 2368)

# AMF matrix (mxn)
amf = np.genfromtxt(r'C:\Users\lme19\Documents\Masters\Validation\O3_1km_SZA_85_95.amf') 

#%%
# SCDs, uncertainties and ancillary data (m)
#date, tmp, sza, balloon, scd, scderr = np.genfromtxt('sunset.scd',skip_header=1, unpack=True) 
# Translate time string into decimal hour [UTC]
#time = [float(str(t)[0:2])+float(str(t)[2:4])/60.+float(str(t)[4:6])/3600. for t in tmp]

# SCDs, uncertainties and ancillary data (m)
date, tmp, sza, balloon, scd, scderr = np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\O3_1km_SZA_85_95.scd',skiprows=2, unpack=True, delimiter = ' ') 
Teilchendichte_cm3, Fehler, mitt_luftdichte_cm3, MischVerh, Fehler, von_km, bis_km =  np.loadtxt(r'C:\Users\lme19\Documents\Masters\Validation\O3_1km_SZA_85_95.vd',skiprows=2, unpack =True) 
#date, tmp, sza, balloon, scd, scderr = np.genfromtxt(r'C:\Users\lme19\Documents\Masters\Validation\Ozone_1km.scd',skip_header=2, skip_footer = 1, unpack=True, delimiter = ' ') 
# Translate time string into decimal hour [UTC]

layer_heights = bis_km - von_km

# 22:00:05 is at index 2368
# need to drop this item from all lists

#error_index = np.where(tmp == 220005)

#scd_clean = np.delete(scd, 2368) + 4.2914745839906196e+18
#scd_clean = abs(scd + 4.2914745839906196e+18)
scd_clean = scd + 3.82752641e+18
#date_clean = np.delete(date, 2368)
#tmp_clean = np.delete(tmp, 2368)
#sza_clean = np.delete(sza, 2368)
#scderr_clean = np.delete(scderr, 2368)

date_clean = date
tmp_clean = tmp
sza_clean = sza
scderr_clean = np.sqrt(scderr**2 + (4.470048173001718e+16)**2)

#added value is SCD error from Fraunhofer offset


# no need to drop the reference spectrum in this case

# But doesn't make that much of a difference - leaving it for now

#fitting to just one region
#scd_clean = scd[3110:3602] + 4.2914745839906196e+18
#date_clean = date[3110:3602]
#tmp_clean = tmp[3110:3602]
#sza_clean = sza[3110:3602]
#scderr_clean = scderr[3110:3602]


#%%

df_scd = pd.read_csv(r'C:\Users\lme19\Documents\Masters\Validation\O3_1km_SZA_85_95.scd', skiprows = 2, delimiter = ' ', dtype = str,  names = ['Date', 'Time/UTC', 'SZA/DEG', 'BALLOON_HEIGHT/KM', 'SCD/MOLEC_CM2', 'SCDERR/MOLEC_CM2'])
#needed to remove the hashtag in front of Date for this to work
#Better to define new column names and leave the hashtag so all files can be read in

#Bit lazy to leave it to manual editing like this 

#Set my own column names now - exactly same as original file but with hashtag removed

# REMEMBER TO CHANGE COLUMN NAMES IF CHANGING EXAMINED SPECIES
        

#date_and_time = df_scd['Date'] + ' ' + df_scd['Time/UTC']
datetime = pd.to_datetime(df_scd['Date'] + ' ' + df_scd['Time/UTC'])

#datetime_clean = datetime.drop(2368)
datetime_clean = datetime
#datetime_clean = datetime[3110:3602]

#%%

# Vertical levels (n+1)
#seg = np.genfromtxt('sunset.seg',skip_header=1) 
# Generate mid-point layers (n)
#height = np.array([seg[i]/2.+seg[i+1]/2 for i in range(len(seg)-1)]) # Height layers (mid-of-levels)

seg = np.genfromtxt(r'C:\Users\lme19\Documents\Masters\Validation\O3_1km_SZA_85_95.seg', delimiter = ' ') 
# Generate mid-point layers (n)
height = np.array([seg[i]/2.+seg[i+1]/2 for i in range(len(seg)-1)]) # Height layers (mid-of-levels)
#all_heights = np.append(height, 80)

# Dimensions
#n = len(all_heights) # parameter vector dimension
n = len(height)
m = len(scd_clean)    # measurement vector dimension

# Define the inverse problem notation
K = amf # forward model matrix
#K = amf_clean
#y = scd_clean # measurement vector
y = scd_clean 
#adding the ref from langley plot
yerr = scderr_clean  # measurment uncertainties


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
ax.errorbar(scd_clean,datetime_clean,xerr=scderr_clean,color='k',marker='.')
ax.set_xlabel('SCD / (molec / cm$^2$)', fontsize=fs)
ax.set_ylabel('Time UTC / h', fontsize=fs)
ax.invert_yaxis()
ax2=ax.twiny()
ax2.plot(sza,datetime,'r-')
ax2.set_xlabel('SZA / $^\circ$', fontsize=fs);



#%%

def tik0_svd(K,y,yerr,alpha):
    """
    # 0th order Tikhonov regularization 
    # via SVD and filter factors
    #
    # arguments: 
    #            K: forward model matrix (m x n) 
    #            y: measurement vector (m)
    #            yerr: measurement uncertainties (m)
    #            alpha: regularization parameter
    #
    # returns: 
    #            x: state estimate (n)
    #            S: a posteriori error covariance matrix (n x n)
    #            A: averageing kernel matrix (n x n)
    """
    m = len(y) # dim. measurements
    n = len(K[0,:]) # dim. state vector
    
    # Consider measurement uncertainties by merging them into K and y
    Kw = np.zeros([m,n])
    yw = np.zeros([m])
    for i in range(m-1):
        Kw[i,:] = K[i,:]/yerr[i] 
        yw[i]   = y[i]/yerr[i]
    
    # Singular value decomposition of forward model
    U,Svec,VT = np.linalg.svd(Kw,full_matrices=False)
    UT = U.T # U transpose
    V = VT.T # V
    F = np.array([s/(s**2+alpha**2) for s in Svec]) # Filter factors / s_i
    
    # Moore Penrose pseudo-inverse with 0th order Tikhonov regularization
    G = V.dot(np.diag(F).dot(UT)) # a.k.a. gain matrix
    # Estimates
    x = G.dot(yw) # state estimate
    S = G.dot(G.T)# a posteriori error covariance
    A = G.dot(Kw) # averaging kernel
   
    return x,S,A


#%%

# Plot estimated O$_3$ profile
fig = plt.figure(figsize=(fheight,fwidth))

# Try out some regularization parameters
alpha = [8E-19,1E-18,1E-17,1E-16,1E-15,2E-15]
for a in alpha:
    xhat,Shat,Ahat = tik0_svd(K,y,yerr,a)
    # Error bars are square-roots of variances
    xhaterr = np.sqrt(np.diag(Shat))
    plt.errorbar(xhat,height,xerr=xhaterr,label=r'Regularization $\alpha$ = %.0e'%a)
plt.ylabel('Height / km', fontsize=fs)
plt.xlabel(r'O$_3$ / molec/cm$^2$', fontsize=fs);
plt.legend();




#%%

# Define a range of alphas for evaluating the solution
alpha = np.arange(np.log(1E-19),np.log(1E-15),0.1) # equidistant in log

# Calculate solutions; residual norm, solution norm;
xhat_L = np.zeros_like(alpha)
res_L = np.zeros_like(alpha)
for i,a in enumerate(alpha):
    xhat,Shat,Ahat = tik0_svd(K,y,yerr,np.exp(a)) # alpha array was defined in log-space, need to undo log
    xhat_L[i] = np.linalg.norm(xhat) # solution norm
    res_L[i] = np.linalg.norm(K.dot(xhat)-y) # residual norm

# Plot L-curve
fig = plt.figure(figsize=(fwidth,fheight))
plt.loglog(res_L,xhat_L,'k');
plt.ylabel(r'Solution norm $|\mathbf{\hat{x}_{\alpha}}|^2$', fontsize=fs)
plt.xlabel(r'Residual norm $|\mathbf{y}-\mathbf{K}\mathbf{\hat{x}_{\alpha}}|^2$', fontsize=fs)
plt.savefig('L_curve_Ozone')


#%%

# Define a rotation operator to turn the L-curve and make the kink a minimum (whose  𝛼 can be easily found).


def rotation_matrix(theta):
    """
    # Rotation matrix for angle theta
    #
    # arguments: 
    #            theta: rotation angle [degree]
    #
    # returns: 
    #            T: rotation matrix (2x2)
    """
    th = theta/180.*np.pi # deg to radians
    T = np.array(( (np.cos(th), -np.sin(th)),(np.sin(th),  np.cos(th)) ))
    return T

# Not sure if this will work 

# My L curve minimum isn't very sharp



#%%

# Rotation matrix for 30 deg
T = rotation_matrix(30)
# Make a vector [residual norm, solution norm]
vec = np.array([[r,x] for r,x in zip(res_L,xhat_L)])
# Rotate the vector
vec_rot = np.array([T.dot(v) for v in vec])
# Plot the rotated L-curve
fig = plt.figure(figsize=(fwidth,fheight))
plt.loglog(vec_rot[:,0],vec_rot[:,1],'k');
plt.ylabel(r'Something weird', fontsize=fs)
plt.xlabel(r'Complementary something weird', fontsize=fs);

#%%

# Find the alpha that corresponds to the minimum of the rotated L-curve
imin = np.argmin(vec_rot[:,1]) # index of array minimum
alpha_L = np.exp(alpha[imin])  # "best" alpha
# Solution for "best" alpha
xhat,Shat,Ahat = tik0_svd(K,y,yerr,alpha_L)
xhaterr = np.sqrt(np.diag(Shat))

#%%

# Plot solution and averaging kernels
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(fwidth,fwidth))
ax1.errorbar(xhat,height,xerr=xhaterr,label=r'Regularization $\alpha$ = %.2e'%alpha_L)
ax1.set_xlabel(r'O$_3$ / molec/cm$^2$', fontsize=fs)
ax1.set_ylabel('Height / km', fontsize=fs);
ax1.legend();

ii = range(1,n,1)
for i in ii:
    ax2.plot(Ahat[i,:],height,label=r'a$_{%d}$'%(i))
ax2.set_xlabel('Averaging kernel', fontsize=fs)
ax2.set_ylabel('Height / km', fontsize=fs);

#%%

# Plot correlation matrix for the "best" alpha
#Chat = np.zeros_like(Shat)
#for i in range(n):
#    for j in range(n):
#        Chat[i,j]=Shat[i,j]/np.sqrt(Shat[i,i])/np.sqrt(Shat[j,j])

#fig, ax = plt.subplots(figsize=(fwidth, fwidth))
#im = plt.imshow(Chat,origin='lower',cmap='RdBu')  
#ax.grid(True)
#ax.figure.colorbar(im, ax=ax, format='% .2f', shrink=0.75);


# section below not really necessary

# Define "worse" regularization parameters
#alpha_worse = [5E-18,1E-17,1E-16,1E-15,2E-15]
#xtrue = xhat


# Plot bias and averaging kernels
#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(fwidth,fwidth))
#ax1.plot(xtrue,height,'k:',label=r'$\mathbf{x_{true}}$')
#for a in alpha_worse:
#    xhat,Shat,Ahat = tik0_svd(K,y,yerr,a)
#    xbias = (Ahat - np.eye(n)).dot(xtrue)
#    p = ax1.plot(xbias,height,label=r'Regularization $\alpha$ = %.0e'%a)
#    c = p[0].get_color()
#    for i in range(1,n,1):
#        ax2.plot(Ahat[i,:],height,color=c,label=r'a$_{%d}$'%(i))
#ax1.set_ylabel('Height / km', fontsize=fs)
#ax1.set_xlabel(r'O$_3$ bias / molec/cm$^2$', fontsize=fs);
#ax1.legend();
#ax2.set_xlabel('Averaging kernel', fontsize=fs)
#ax2.set_ylabel('Height / km', fontsize=fs);

#%%


# Converting to molecules / cm3
# Might be worth cutting off SZAs above 95 to improve results? Very large errors and don't seem realistic anymore

height_cm = layer_heights * 100000




#%%

# xhat gives molecules / cm2
# layer height is in cm
# mitt lufdichte is molecules /cm3

conc = xhat / height_cm

conc_err = xhaterr / height_cm

vmr = conc / mitt_luftdichte_cm3

vmr_err = conc_err / mitt_luftdichte_cm3

plt.plot(conc[10:36], height[10:36])
plt.xlabel('O3 Concentration in molecules/cm3')


#%%

plt.plot(vmr[10:36], height[10:36])
plt.xlabel('VMR')
plt.ylabel('height')
plt.show()

#%%


# Plot solution and averaging kernels
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(fwidth,fwidth))
ax1.errorbar(conc,height,xerr=conc_err,label=r'Regularization $\alpha$ = %.2e'%alpha_L)
ax1.set_xlabel(r'O$_3$ / molec/cm$^3$', fontsize=fs)
ax1.set_ylabel('Height / km', fontsize=fs);
ax1.legend();

ii = range(1,n,1)
for i in ii:
    ax2.plot(Ahat[i,:],height,label=r'a$_{%d}$'%(i))
ax2.set_xlabel('Averaging kernel', fontsize=fs)
ax2.set_ylabel('Height / km', fontsize=fs);

plt.savefig('Ozone_profile_with_kernel')

#%%

# VMR plot

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(fwidth,fwidth))
ax1.errorbar(vmr[10:36],height[10:36],xerr=vmr_err[10:36],label=r'Regularization $\alpha$ = %.2e'%alpha_L)
ax1.set_xlabel(r'O3 / Volume Mixing Ratio', fontsize=fs)
ax1.set_ylabel('Height / km', fontsize=fs);
ax1.legend();

ii = range(10,36,1)
for i in ii:
    ax2.plot(Ahat[i,:][10:36],height[10:36],label=r'a$_{%d}$'%(i))
ax2.set_xlabel('Averaging kernel', fontsize=fs)
ax2.set_ylabel('Height / km', fontsize=fs);

plt.savefig('O3_profile_with_kernel_zoomed_vmr')


#%%

# Validation


#from pysolar.solar import *
#Input Path for Standard Atmosphere
PATH_atm = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\StandardAtm1976.txt'
#Input Path for ECMWF data
PATH_era5 = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\20230111_downloadERA5_Geopot-Temp-Ozone.nc'


O3_Timmins = xr.open_dataset(r'C:\Users\lme19\Documents\Masters\Validation\MLS_O3_L3_Aug2022_netcdf\MLS-Aura_L3DB-O3_v05-02-c01_2022d235.nc', group='O3 PressureGrid')


O3_Timmins_Aug22 = O3_Timmins.sel(time="2022-08-23", lat="48", lon="-82.5")


pressure_vals = O3_Timmins_Aug22.lev

vmr_vals = O3_Timmins_Aug22.value


# Convert pressure into vmr

# using standard atmosphere file

heights, temps, pressures = np.loadtxt(PATH_atm,skiprows=2,unpack=True)

pressure = np.append(pressures, 1.000000e-05)
height_val = np.append(heights, 120)
temp = np.append(temps, 398)


#%%

from scipy import interpolate
f = interpolate.interp1d(pressure, height_val)

xnew = np.arange(0.1, 1000, 10)
ynew = f(xnew)   # use interpolation function returned by `interp1d`
plt.plot(pressure, height_val, 'o', xnew, ynew, '-')
plt.show()

heights_MLS = f(pressure_vals)

#%%

#height_vals_mls = func(height, *popt),

plt.errorbar(vmr[10:36], height[10:36],  xerr = vmr_err[10:36], label = 'O3 retrieved', marker = 'x')
plt.plot(vmr_vals[10:30], heights_MLS[10:30], label = 'O3 MLS')
plt.xlabel('VMR')
plt.ylabel('Altitude (km)')
plt.legend()
plt.grid()
plt.savefig('MLS Validation')
plt.show()


#%%

#Sonde Validation Data

dataset = np.genfromtxt(r'C:\Users\lme19\Documents\Masters\Validation\o3_20220824_0430_formated.txt', skip_header = 2)


HeightMSL = dataset[:,1] 

total_pressure = dataset[:,2] * 100000
# gives total pressure in mPa 
#just like ozone pressure

O3 = dataset[:,7]

O3_current = dataset[:,9]

Integrated_ozone = dataset[:,10]

Residual_ozone = dataset[:,11]

O3_voltage = dataset[:,12]

O3_aux = dataset[:,13]

O3_pump_current = dataset[:,14]

height_km = HeightMSL * 0.001

VMR_vals = O3 / total_pressure

plt.errorbar(VMR_vals, height_km, label = 'O3 Sonde')
plt.errorbar(vmr[0:20], height[0:20], xerr = vmr_err[0:20], label = 'O3 retrieved', marker = 'x')
plt.xlabel('Volume Mixing Ratio')
plt.ylabel('Altitude (km)')
plt.legend()
plt.grid()
plt.savefig('Sonde_validation_data', dpi = 300)
plt.show()


# Sonde data doesn't seem to go past 20km

#%%


plt.errorbar(vmr[10:36], height[10:36],  xerr = vmr_err[10:36], label = 'O3 retrieved')
plt.plot(VMR_vals, height_km, label = 'O3 Sonde')
plt.plot(vmr_vals[10:30], heights_MLS[10:30], label = 'O3 MLS')
plt.xlabel('VMR')
plt.ylabel('height')
plt.legend()
plt.grid()
plt.savefig('MLS amd Sonde Validation')
plt.show()

# The errors are on x not y you numpty











