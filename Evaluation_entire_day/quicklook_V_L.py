# -*- coding: utf-8 -*-
"""
This file is used as an ORIGIN alternative.
You can adjust the CONSTANT NAMES to your AutomatischeAuswertung output names and 
cross section file names (which are the column names in the ASCII output). 
You'll get a quicklook on the data w/o the need to use origin.
You're welcome to use this script as a starting point for your evaluation, but please preserve it here!
"""
#%% IMPORTS
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import numpy as np
sns.set_style('darkgrid')
#%% CONSTANT NAMES
NO2FILE = r"C:\Users\FPDOAS\Documents\FP\aktuell\no2res.dat"
O3FILE = r"C:\Users\FPDOAS\Documents\FP\FP-Praktikanten_innen\2023_02_20_Livia_Vincent\Evaluation_entire_day\ozonres.dat"


NO2XSNAME = r"convolution_result_NO2"
O3XSNAME = r"convolution_result_O3"
#%% LOAD DATA
no2data = pd.read_csv(NO2FILE, sep='\t')
o3data = pd.read_csv(O3FILE, sep='\t')

no2data.columns = no2data.columns.str.strip()
o3data.columns = o3data.columns.str.strip()
#%% NO2
no2data['timestamp'] = no2data['Datum']+no2data['StartZeit']
no2data['timestamp'] = no2data['timestamp'].apply(pd.to_datetime, format='%d.%m.%Y%H:%M:%S')
plt.figure('NO2')
plt.title('NO2 dSCD over day')
plt.errorbar(no2data['timestamp'], no2data[f'{NO2XSNAME}'], yerr=no2data[f'{NO2XSNAME} Error'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('time')
plt.ylabel('NO2 dSCD')
plt.savefig('NO2_dSCD.png')
plt.show()
#%% O3
o3data['timestamp'] = o3data['Datum']+o3data['StartZeit']
o3data['timestamp'] = o3data['timestamp'].apply(pd.to_datetime, format='%d.%m.%Y%H:%M:%S')
plt.figure('O3')
plt.title('O3 dSCD over day')
plt.errorbar(o3data['timestamp'], o3data[f'{O3XSNAME}'], yerr=o3data[f'{O3XSNAME} Error'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xlabel('time')
plt.ylabel('O3 dSCD')
plt.savefig('O3_dSCD.png')
plt.show()

#%%




























