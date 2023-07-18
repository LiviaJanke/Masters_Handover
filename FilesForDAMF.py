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

#from pysolar.solar import *


#%%
#Define functions

def readTrajetory(PATH):
    """
    

    Parameters
    ----------
    PATH : TYPE
        DESCRIPTION.

    Returns
    -------
    trajectory : TYPE
        DESCRIPTION.

    """
    data = np.loadtxt(PATH, skiprows=3, usecols=(3, 4, 5), delimiter="\t")
    time_raw = np.loadtxt(PATH, skiprows=3, usecols=(0), delimiter="\t", dtype=str)

    time = []
    for i in time_raw:
        dummy = datetime.strptime(i, "%Y-%m-%d %H:%M:%S")
        dummy = dummy.replace(tzinfo=timezone.utc)
        time.append(dummy.timestamp())

    trajectory = np.vstack((time, data[:, 0], data[:, 1], data[:, 2]))
    return trajectory
#returns time, GPS height, pressure , GasTemp






#%%
#Define Paths

#Input Path for Trajectory data
#PATH_tr =  r'C:/Users/Karolin Voss/Documents/Data/Timmins2022/lvl1/Trajectory_Flight2022_lvl1.csv'
#Input Path for QDOAS OutputFile
#PATH_qdoas = r'C:/Users/Karolin Voss/Documents/PhD_Karolin/Evaluation/2022TimminsBalloon/DOAS_Evaluation/20221214_NoGases_Ascent+Sunset_200coadd.ASC'
#Input Path for Standard Atmosphere
PATH_atm = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\StandardAtm1976.txt'
#Input Path for ECMWF data
PATH_era5 = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\20230111_downloadERA5_Geopot-Temp-Ozone.nc'
#Input Path from Sondage data
#PATH_sonde = r"C:/Users/Karolin Voss/Documents/Data/Timmins2022/CNESdata/Flight data/23082022/sondage_20220823_1833.txt"
#Input Path from GLORIA data
#PATH_gloria = r"C:/Users/Karolin Voss/Documents/Data/Timmins2022/GLORIAdata/Gloria_pressure_altitude.txt"

PATH_kiruna = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\Trajectory_Flight2022_lvl1.csv'

PATH_QDOAS = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\RingSpectrum.asc'

#%% settings
#Output Basepath
PATH_out = r'C:\Users\lme19\Documents\damf_butz\files_for_damf\IO_1km_SZA_85_95'
#Prefix for output data
out_prefix = r'IO_1km_SZA_85_95'

filewrite = True
cut = 62820 # Timmins # Kiruna: 62820 #cut data after cut of balloon

#time of flight and other parameters
time_era5 = 17
lat_Timmins = 48.47
long_Timmins = -81.33
g= 9.81

noGases = True

#using era5 data so this is correct



#%%
#Read QDOAS Data
if noGases:
    qdata_plain = pd.read_csv(PATH_QDOAS,skiprows = 8, delimiter = '\t') #VIS 8 ###
    #qdata = pd.read_csv(PATH_QDOAS,skiprows = 8, delimiter = '\t') #VIS 8 ###
else:
    qdata_plain = pd.read_csv(PATH_QDOAS,header=8,delimiter='\t') #VIS 8 ###
    #qdata = pd.read_csv(PATH_QDOAS,header=8,delimiter='\t') #VIS 8 ###
qdataC = qdata_plain.columns

# Add some stuff in here to strip out the reference spectrum etc?
# yep seems this would be the spot

df_no_reference = qdata_plain.drop([2368])

df_high_scans = df_no_reference[df_no_reference.Scans > 30]

#Choose a set of the dataframe for SZA greater than 90? 

df_high_SZA = df_high_scans[df_high_scans.SZA > 85]

df_constrained_SZA = df_high_SZA[df_high_SZA.SZA < 95]

qdata = df_constrained_SZA.reset_index(drop=True)

#indexes = qdata.index.values.tolist()

#convert to dateime
qtime = []
for i in range(len(qdata['Time (hh:mm:ss)'])):
#for i in indexes:
    dummy = (qdata['# Date (DD/MM/YYYY)'][i] + ' ' + qdata['Time (hh:mm:ss)'][i]) #VIS 0,1
    if len(dummy)>20:
        dtime = datetime.strptime(dummy, '%d/%m/%Y %H:%M:%S.%f')
        qtime.append(dtime.replace(tzinfo=timezone.utc))
    else:
        dtime = datetime.strptime(dummy, '%d/%m/%Y %H:%M:%S')
        qtime.append(dtime.replace(tzinfo=timezone.utc))
del dummy,i
#%% Read Trajectory Data
# time,height,pressure,Temp
tdata = readTrajetory(PATH_kiruna)
tdata = tdata[:,1:]
#%% Read Standard Atmosphere Data
atmdata = np.loadtxt(PATH_atm,skiprows=2,unpack=True)
# read ECMWF data
DS = xr.open_dataset(PATH_era5)
DS = DS.assign(h = DS.z/g)
DS.h.attrs['units']= 'm'
DS.h.attrs["standard_name"] = "height"
DS.h.attrs["long_name"] = "Height"
# DS.t = DS.t.values - 273.15
# DS.t.attrs["units"] = 'K'
# read sonde data
#index, press_sond, height_sond, temp_sond, unknown1_sond, unknown2_sond = np.loadtxt(PATH_sonde, unpack=True)
# read GLORIA data
#timestamp_GLORIA, press_gloria, h_gloria = np.loadtxt(PATH_gloria, unpack=True, skiprows=1, delimiter=';')
#%% plot pressure profiles from different datasets:
plt.figure()
# DS_mean_ascent.plot(y = "level")
DS.h.isel(time=[17], longitude=[1], latitude=[1]).plot(label='ERA5', x= 'level', hue='time', color = 'blue')
plt.plot(tdata[2][:cut],tdata[1][:cut],label='Balloon trajectory CNES',color='r')
plt.plot(atmdata[2],atmdata[0]*1000,label='US Standard 1976',color='k')
#plt.plot(press_sond, height_sond, label='sondage CNES 20220823_1833', color='orange')
#plt.plot(press_gloria, h_gloria, label='GLORIA', color='gray')
plt.xscale('log')
plt.grid()
plt.legend()

#%% plot temp profiles from different datasets
# temp_era5 = DS.t.isel(time=[17], longitude=[1], latitude=[1])
plt.figure('TempProfile')
plt.plot(atmdata[1],atmdata[0]*1000,label='US Standard 1976',color='k')
plt.plot(tdata[3][:cut] + 273.15,tdata[1][:cut],label='Balloon trajectory CNES',color='r')
#plt.plot(temp_sond + 273.15, height_sond, label='sondage CNES 20220823_1833', color='orange')
DS.isel(time=[17], longitude=1, latitude=1).plot.scatter(marker = '.', label='ERA5',y='h', x='t', color = 'blue')
plt.grid()
plt.legend()
# plt.xlabel('Temp in K')
# plt.ylabel('Height in m')
# plot measured vs standard atmosphere
if False:
    # plt.figure('TempProfile')
    # plt.plot(atmdata[1]-273.15,atmdata[0]*1000,label='US Standard 1976',color='k')
    # plt.plot(tdata[3][:cut],tdata[1][:cut],label='Measured',color='r')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Temp in °C')
    # plt.ylabel('Height in m')
    
    # plt.figure('PressureProfile')
    # plt.plot(atmdata[2],atmdata[0]*1000,label='US Standard 1976',color='k')
    # plt.plot(tdata[2][:cut],tdata[1][:cut],label='Measured',color='r')
    # plt.grid()
    # plt.legend()
    # plt.xscale('log')
    # plt.xlabel('TPressure in mbar')
    # plt.ylabel('Height in m')
    # del cut
    """ERA5 seems best profile for altitudes lower than 48 km, then no more information from ERA5, 
    thus scale US standard atmosphere to ERA5 at that height and use scaled US standard above 48 km"""
#%% scale std atmo to era5 data
height_ERA5_Timmins = DS.h.isel(time=time_era5, longitude=1, latitude=1)
hmax = (np.max(height_ERA5_Timmins)).values #maximal height of ERA5 data
# interpolation funtion of std atmo
height2press_std = interp1d(atmdata[0]*1000, np.log(atmdata[2]), kind="linear")
height2temp_std = interp1d(atmdata[0]*1000, atmdata[1], kind="linear")

press_std_scalingFactor = height_ERA5_Timmins.level[0].values / np.exp(height2press_std(hmax))
temp_std_scalingFactor = DS.t.isel(time=[time_era5], longitude=[1], latitude=[1])[0,0,0,0].values / height2temp_std(hmax)

#%% Write Atmospheric Profile in atm File
k = 120 #highest level to calculate in km, needs to be less or eqal than max given in standard atm
#Standard atm has a max of 120
inc = 1 #increments in km
height_damf = np.round(np.linspace(1,k,int((k-1)*1/inc)+1),3)

#calculate interpolation functions
height2press_low = interp1d(height_ERA5_Timmins.values/1000, np.log(height_ERA5_Timmins.level.values), kind="linear")
height2temp_low = interp1d(np.flip(height_ERA5_Timmins.values/1000), np.flip(DS.t.isel(time=time_era5, longitude=1, latitude=1).values), kind="linear")

# height2press_high = interp1d(atmdata[0], atmdata[2]*press_std_scalingFactor, kind="linear")
height2press_high = interp1d(atmdata[0], np.log(atmdata[2]*press_std_scalingFactor), kind="linear")
height2temp_high = interp1d(atmdata[0], atmdata[1]*temp_std_scalingFactor, kind="linear")

press_damf = np.zeros_like(height_damf)
temp_damf= np.zeros_like(height_damf)
for i in range(len(height_damf)):
    if height_damf[i] < hmax/1000:
        print(height_damf[i])
        press_damf[i] = np.exp(height2press_low(height_damf[i]))
        temp_damf[i] = height2temp_low(height_damf[i])
    elif height_damf[i] >= hmax/1000:
        print(height_damf[i])
        # press_damf[i] = height2press_high(height_damf[i])
        press_damf[i] = np.exp(height2press_high(height_damf[i]))
        temp_damf[i] = height2temp_high(height_damf[i])

if filewrite: 
    f = open(PATH_out + '\\' + out_prefix + '.atm','w')
    #write header
    f.write('# Profile Data from ERA5, above ' + str(hmax) +' km interpolated to US Standard Atmosphere 1976' + '\n')
    f.write('# Height/km Temperature/K Pressure/mbar'+ '\n')
    
    f.write(str(atmdata[0][0]) + ' ' + str(np.round(atmdata[1][0],3)) + ' ' + str(np.round(atmdata[2][0],3)) + '\n')
    for i in range(len(height_damf)):
        f.write(str(height_damf[i]) + ' ' + str(np.round(temp_damf[i],3)) + ' ' + str(np.round(press_damf[i],8)) + '\n')

    f.close()
    del f, i

#%% Write Atmospheric Profile in atm File Holzbeck version
   
# k = 120 #highest level to calculate in km, needs to be less or eqal than max given in standard atm
# inc = 0.1 #increments in km

# hmax = 35 #maximal height for interpolation from data in km; e.g. below float

# #calculate interpolation functions
# height2press_low = interp1d(tdata[1][:cut]/1000, tdata[2][:cut], kind="linear")
# height2temp_low = interp1d(tdata[1][:cut]/1000, tdata[3][:cut] + 273.15, kind="linear")

# height2press_high = interp1d(atmdata[0], atmdata[2], kind="linear")
# height2temp_high = interp1d(atmdata[0], atmdata[1], kind="linear")

# height = np.linspace(1,k,int((k)*1/inc))

# #calculate interpolatet temp and pres profile
# temp = []
# press = []
# for h in height:
#     if h <= hmax:
#         press.append(height2press_low(h))
#         temp.append(height2temp_low(h))
#     else:
#         press.append(height2press_high(h))
#         temp.append(height2temp_high(h))

# if filewrite: 
#     f = open(PATH_out + '\\' + out_prefix + '.atm','w')
#     #write header
#     f.write('# Profile Data from measurement, above ' + str(hmax) +' km interpolated to US Standard Atmosphere 1976' + '\n')
#     f.write('# Height/km Temperature/K Pressure/mbar'+ '\n')
    
#     f.write(str(atmdata[0][0]) + ' ' + str(np.round(atmdata[1][0],3)) + ' ' + str(np.round(atmdata[2][0],3)) + '\n')
#     for i in range(len(height)):
#         f.write(str(height[i]) + ' ' + str(np.round(temp[i],3)) + ' ' + str(np.round(press[i],5)) + '\n')

#     f.close()
#     del f, i
# del h,k,cut,inc,hmax,height2press_high,height2press_low,height2temp_low,height2temp_high

#%% plot damf profiles and data that it was constructed from
if True:
    cut = 62820 # Timmins # Kiruna: 62820 #cut data after cut of balloon
    plt.figure('TempProfileDamf')
    plt.plot(atmdata[1]*temp_std_scalingFactor,atmdata[0]*1000,label='US Standard 1976',color='k')
    # plt.plot(tdata[3][:cut]+273.15,tdata[1][:cut],label='Measured',color='r')
    plt.plot(temp_damf, height_damf*1000, label='interpolated profile', marker='+', markersize=4)
    DS.isel(time=17, longitude=1, latitude=1).plot.scatter(marker = '.', label='ERA5',y='h', x='t', color = 'blue')
    # plt.plot(np.flip(DS.t.values[0,:,0,0]), np.flip(height_ERA5_Timmins.values[0,:,0,0]), color='orange', label='ERA5_2')
    plt.grid()
    plt.legend()
if True:
    plt.figure('PressureProfileDamf_nolog')
    plt.plot(atmdata[2]*press_std_scalingFactor,atmdata[0]*1000,label='US Standard 1976',color='k')
    # plt.plot(tdata[2][:cut],tdata[1][:cut],label='Measured',color='r')
    DS.h.isel(time=[time_era5], longitude=[1], latitude=[1]).plot(label='ERA5', x= 'level', hue='time', color = 'blue')
    plt.plot(press_damf, height_damf*1000, label='interpolated profile')
    plt.grid()
    plt.legend()
    plt.xscale('log')
    #%% Match GPS height to QDOAS data
time2height = interp1d(tdata[0], tdata[1], kind="linear")

qheight = np.zeros(len(qtime))
for i in range(len(qtime)):
    qheight[i] = time2height(qtime[i].replace(tzinfo=timezone.utc).timestamp())
    
del time2height
#%% plot interpolated height and original height
qtime_ts = [t.replace(tzinfo=timezone.utc).timestamp() for t in qtime]

if True:
    plt.figure()
    plt.plot(tdata[0], tdata[1], label='original')
    plt.plot(qtime_ts, qheight, label='interp')
    plt.legend()
    plt.grid()
#%% Write Segmets File
if filewrite:
    step = 1 #chosse step size in km
    
    # also need to set resolution here
    
    
    
    MAX = 80 #choose maximal height in km
    f = open(PATH_out + '\\' + out_prefix + '.seg','w')
    i = 0 
    while i <= MAX:
        i = np.round(i,4)
        f.write(str(i)+ '\n')
        i = i + step
    f.close()
    del f,i,MAX
#%% Write SCD File
if filewrite:
    #choose species by column index of qdataC to write SCD
    
    # 8 seems to be the index for ozone
    # IO has index 34
    # Error has index 35
    # how to incorporate error here?
    
    # NO2 has index 10
    
    # Run DAMF for the error?
    
    # Run for NO2 as wll - index is 10
    
    if noGases:
        k = 34 ###
    else:
        k = 34
    
    #mask SCD data for SZA smaller than 80 (90)   
    mask = np.array(qdata['SZA']) > 1 #all true for langley #UV 5 VIS3
    qtime_masked = np.array(qtime)[mask]
    
    f = open(PATH_out + '\\' + out_prefix + '.scd','w')
    #write header
    f.write('# ' + str(qdataC[k]) + '\n')
    f.write('# Date Time/UTC SZA/DEG BALLOON_HEIGHT/KM SCD/MOLEC_CM2 SCDERR/MOLEC_CM2' + '\n')
    
    for i in range(len(qtime_masked)):
        day = str(qtime_masked[i].day)
        if len(day) < 2:
            day = '0' + day
            
        month = str(qtime_masked[i].month)
        if len(month) < 2:
            month = '0' + month
            
        hour = str(qtime_masked[i].hour)
        if len(hour) < 2:
            hour = '0' + hour
            
        minute = str(qtime_masked[i].minute)
        if len(minute) < 2:
            minute = '0' + minute
            
        second = str(int(qtime_masked[i].second))
        if len(second) < 2:
            second = '0' + second
            
        #for scd with 2 digid exponent
        scd = str(qdata[mask][qdataC[k]][i])#str(qdata[mask][qdataC[k]][i])[:-5] + 'E' + str(qdata[mask][qdataC[k]][i])[-2:]
        scd_err = str(qdata[mask][qdataC[k+1]][i])#str(qdata[mask][qdataC[k+1]][i])[:-5] + 'E' + str(qdata[mask][qdataC[k+1]][i])[-2:]
        # errors are aready included here
        # nice
        
        
        f.write(day + month + str(qtime_masked[i].year)[-2:] + ' ' + hour + minute 
                + second + ' ' + str(np.round(qdata[mask]['SZA'][i],4)) + ' ' #UV 5 VIS 3
                + str(qheight[i]/1000) + ' ' + scd + ' ' + scd_err + '\n' )
    
    f.close()
    del day, month, hour, minute, second, scd, i, k, f
#%%
#create Trajectory .bzp File
if filewrite:
    #needs same mask created in writing scd file
    
    f = open(PATH_out + '\\' + out_prefix + '.bzp','w')
    #write header
    f.write('# Date Time/UTC LONGITUDE/DEG LATITUDE/DEG BALLOON_HEIGHT/KM' + '\n')
    
    for i in range(len(qtime_masked)):
        day = str(qtime_masked[i].day)
        if len(day) < 2:
            day = '0' + day
            
        month = str(qtime_masked[i].month)
        if len(month) < 2:
            month = '0' + month
            
        hour = str(qtime_masked[i].hour)
        if len(hour) < 2:
            hour = '0' + hour
            
        minute = str(qtime_masked[i].minute)
        if len(minute) < 2:
            minute = '0' + minute
            
        second = str(int(qtime_masked[i].second))
        if len(second) < 2:
            second = '0' + second
        
        f.write(day + ' ' + month + ' ' + str(qtime_masked[i].year) + ' ' + hour + ' ' 
                + minute + ' ' + second + ' ' + str(np.round(qdata['Longitude'][i],5)) + ' ' #vis 4
                + str(np.round(qdata['Latitude'][i],5)) + ' ' + str(qheight[i]/1000) + '\n')    #vis 5
            
    f.close()
    del day, month, hour, minute, second, i, f
#%% create SZA .sza file

qsza = np.zeros(len(qdata))
# calculate half the time a spectrum takes and add it to the start_time to get the middle time of each spec qtime_corr
dt = np.array([timedelta(seconds=0.5*qdata["Tint"][i]*qdata["Scans"][i]) for i in range(len(qdata))])
qtime_corr = qtime + dt
# iterate through each spectrum to calculate the astronmomical SZA
for i in range(len(qdata)):
    alt = get_altitude(qdata["Latitude"][i], qdata["Longitude"][i], qtime_corr[i], pressure=0)
    qsza[i] = 90 - alt

# write .sza file
f = open(PATH_out + '\\' + out_prefix + '.sza','w')
#write header
# f.write('# SZA calculated with pysolar, pressure=0 for each spectrum\n')
# f.write('# Date \t Time \t SZA/°'+ '\n')

for i in range(len(qdata)):
    f.write(qtime_corr[i].strftime("%d.%m.%Y") + '\t' + qtime_corr[i].strftime("%H:%M:%S.%f") + '\t' + str(np.round(qsza[i], 4)) + '\n')

f.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
