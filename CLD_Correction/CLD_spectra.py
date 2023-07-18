# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:14:22 2021

@author: kvoss
"""

# import heidoas_v1_3 as doas
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

#%%
# declare paths for disk and center spectra
# files 01-10 are disk spectra and 11-20 are center spectra
path_disk = r'C:\Users\lme19\Documents\CLD_Correction\FTS-Atlas\file02'
path_center = r'C:\Users\lme19\Documents\CLD_Correction\FTS-Atlas\file12'

# read in spectra
wvl_disk, spectrum_disk, continuumInt_disk = np.loadtxt(str(path_disk), unpack=True, dtype = "float")
wvl_center, spectrum_center, continuumInt_center = np.loadtxt(str(path_center), unpack=True, dtype = "float")
# plot spectra
plt.figure('HR disk and center spectrum')
plt.plot(wvl_disk, spectrum_disk, label = 'disk')
plt.plot(wvl_center, spectrum_center, label = 'center')
plt.legend()
plt.xlabel('wavelength / Angstrom')
plt.ylabel('intensity')

# change wavelength to nm
wvl_disk = wvl_disk/10
wvl_center = wvl_center/10

print(len(wvl_center))
print(len(wvl_disk))

spectrum_disk =np.interp(wvl_center,wvl_disk,spectrum_disk)
print(len(spectrum_disk))
#%%


#Convolve with Gaussian Function
#wvl disk has length of 119581

#def gaus(X,C,X_mean,sigma):
#   return C*exp(-(X-X_mean)**2/(2*sigma**2))

path_slf_Kr = r'C:\Users\lme19\Documents\CLD_Correction\VIS_slf_Kr432.slf'
path_slf_Hg = r'C:\Users\lme19\Documents\CLD_Correction\VIS_slf_Hg.slf'

slf_Kr, vals_Kr = np.loadtxt(str(path_slf_Kr), unpack=True, dtype = "float")
#what is the second part in these slit functions?
#second part is the actual slit func - first is the x-axis values
#don't the x-axis values need to be taken into account too somehow?

slf_Hg, vals_Hg = np.loadtxt(str(path_slf_Hg), unpack=True, dtype = "float")


#%%

plt.plot(slf_Hg, label = 'slf')
plt.plot(vals_Hg, label = 'vals')
plt.legend()
plt.show()


plt.plot(slf_Hg, vals_Hg)



#%%

conv1 = np.convolve(spectrum_disk, vals_Kr, 'same')
print(len(conv1))

conv2 = np.convolve(spectrum_center, vals_Kr, 'same')
print(len(conv2))




sigma_Kr = (conv1 - conv2) / conv2
#this doesn't work because wvl_disk and wvl_center have different values
#why is this??? Doesn't really make sense?

print(len(sigma_Kr))

#Try interpolating one dataset onto the other?

#%%

plt.plot(wvl_center, spectrum_disk)
plt.plot(wvl_center, conv1, label = 'conv1')
plt.legend()



 #%%

plt.plot(wvl_center, conv1, label = 'C')


plt.plot(wvl_center, sigma_Kr)


#this seems to be the CLD spectrum
#now to integrate this into DOAS
#what kind of file do I want to write it into?

#%%


wvl_and_sigma = np.stack((wvl_center, sigma_Kr), axis = -1)


#%%

np.savetxt('CLD_sigmas.xs', wvl_and_sigma, delimiter ='\t')

#%%

#CLD 

#center = center_hr.convolute(cslf, grid = calib.calibration[310:]) #, column = 1e17, fraunhofer = fraunhofer_hr) 

#disk = disk_hr.convolute(cslf, grid = calib.calibration[310:]) #, column = 1e17, fraunhofer = fraunhofer_hr) 

  

# CLD = center.copy() 

# intensity_CLD = (disk.intensity - center.intensity)/disk.intensity  

# CLD.intensity = intensity_CLD # - np.nanmean(intensity_CLD) 

# CLD.name = 'CLD' 

  

# plt.figure('CLD crosssection') 

# plt.plot(CLD.wavelength, CLD.intensity, label='CLD') 

# plt.plot(center.wavelength, center.intensity, label='center') 

# plt.plot(disk.wavelength, disk.intensity, label='disk') 

# plt.legend() 

#%%

path_disk = r'C:\Users\lme19\Documents\CLD_Correction\FTS-Atlas\file02'
path_center = r'C:\Users\lme19\Documents\CLD_Correction\FTS-Atlas\file12'

# read in spectra
wvl_disk, spectrum_disk, continuumInt_disk = np.loadtxt(str(path_disk), unpack=True, dtype = "float")
wvl_center, spectrum_center, continuumInt_center = np.loadtxt(str(path_center), unpack=True, dtype = "float")

wvl_disk = wvl_disk/10
wvl_center = wvl_center/10

wavelength_spectrum_file02 = np.stack((wvl_disk, spectrum_disk) , axis = -1)

np.savetxt(r'C:\Users\lme19\Documents\CLD_Correction\wavelength_spectrum_file02.xs', wavelength_spectrum_file02, delimiter =' ')


wavelength_cen_spectrum_file02 = np.stack((wvl_center, spectrum_center) , axis = -1)

np.savetxt(r'C:\Users\lme19\Documents\CLD_Correction\wavelength_cen_spectrum_file12.xs', wavelength_cen_spectrum_file02, delimiter =' ')


#%%

#re-doing with the DOAS convolutions


wav1, conv1 = np.loadtxt(r"C:\Users\lme19\Documents\CLD_Correction\convolution_center.xs", skiprows = 12 , delimiter = ' ', unpack = True)

wav2, conv2 = np.loadtxt(r"C:\Users\lme19\Documents\CLD_Correction\convolution_disk.xs", skiprows = 12 , delimiter = ' ', unpack = True)

#%%

sigma_Kr = (conv1 - conv2) / conv2


#%%

wvl_disk = wav1/10
wvl_center = wav2/10

wvl_and_sigma = np.stack((wvl_center, sigma_Kr), axis = -1)



np.savetxt(r'C:\Users\lme19\Documents\CLD_Correction\CLD_sigmas_v2.xs', wvl_and_sigma, delimiter ='\t')





































