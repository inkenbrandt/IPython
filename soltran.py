# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import glob
import re
import xmltodict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.ticker as tick


def new_xle_imp(infile):
    '''
    This function uses an exact file path to upload a Solinst xle file. 
    
    infile = complete file path to input file
    
    RETURNS
    A pandas dataframe containing the transducer data
    '''
    # open text file
    with open(infile) as fd:
        # parse xml
        obj = xmltodict.parse(fd.read(),encoding="ISO-8859-1")
    # navigate through xml to the data
    wellrawdata = obj['Body_xle']['Data']['Log']
    # convert xml data to pandas dataframe
    f = pd.DataFrame(wellrawdata)
    # get header names and apply to the pandas dataframe
    f[str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f['ch2']
    
    tempunit = str(obj['Body_xle']['Ch2_data_header']['Unit'].encode("ISO-8859-1"))
    if tempunit == 'Deg C' or tempunit == '\xb0C':
        f[str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f['ch2']
    elif tempunit == 'Deg F' or tempunit == '\xb0F': 
        f[str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f['ch2']*0.33456
    else:
        f[str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f['ch2']
        print "Unknown Units"
    
    levelunit = str(obj['Body_xle']['Ch1_data_header']['Unit']).lower()
    if levelunit == "feet" or levelunit == "ft":
        f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']
    elif levelunit == "kpa":
        f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']*0.33456
    elif levelunit == "mbar":
        f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']*0.0334552565551
    elif levelunit == "psi":
        f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']*2.306726
    else:
        f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']
        print "Unknown Units"
    # add extension-free file name to dataframe
    f['name'] = getfilename(infile)
    # combine Date and Time fields into one field
    f['DateTime'] = pd.to_datetime(f.apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
    f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()].convert_objects(convert_numeric=True)
    f[str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f[str(obj['Body_xle']['Ch2_data_header']['Identification']).title()].convert_objects(convert_numeric=True)
    f = f.reset_index()
    f = f.set_index('DateTime')
    f = f.drop(['Date','Time','@id','ch1','ch2','index','ms'],axis=1)
    return f

    
def getfilename(path):
    ''' this function extracts the file name without file path or extension '''
    return path.split('\\').pop().split('/').pop().rsplit('.', 1)[0]
    
def hourly_resample(df,bse=0,minutes=60):
    '''
    INPUT
    -----
    df = pandas dataframe containing time series needing resampling
    bse = base time to set; default is zero (on the hour); 
    minutes = sampling recurrance interval in minutes; default is 60 (hourly samples)
    
    RETURNS
    -----
    A pandas dataframe that has been resampled to every hour, at the minute defined by the base (bse)
    
    DESCRIPTION
    -----
    see http://pandas.pydata.org/pandas-docs/dev/generated/pandas.DataFrame.resample.html for more info
    
    This function uses pandas powerful time-series manipulation to upsample to every minute, then downsample to every hour, 
    on the hour.
    
    This function will need adjustment if you do not want it to return hourly samples, or if you are sampling more frequently than
    once per minute.
    
    see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    
    '''
    df = df.resample('1Min') #you can make this smaller to accomodate for a higher sampling frequency
    df = df.interpolate(method='time') #http://pandas.pydata.org/pandas-docs/dev/generated/pandas.Series.interpolate.html
    df = df.resample(str(minutes)+'Min', how='first',closed='left',label='left', base=bse) #modify '60Min' to change the resulting frequency
    return df

def fcl(df, dtObj):
    '''
    finds closest date index in a dataframe to a date object
    
    df = dataframe
    dtObj = date object
    
    taken from: http://stackoverflow.com/questions/15115547/find-closest-row-of-dataframe-to-given-time-in-pandas
    '''
    return df.iloc[np.argmin(np.abs(df.index.to_pydatetime() - dtObj))]
   

def baro_drift_correct(wellfile,barofile,manualfile,sampint=60,wellelev=4800,stickup=0):
    '''
    INPUT
    -----
    wellfile = pandas dataframe with water level data labeled 'Level'; index must be datetime
    barofile = pandas dataframe with barometric data labeled 'Level'; index must be datetime
    manualfile = pandas dataframe with manual level data in the first column after the index; index must be datetime
    
    sampint = sampling interval in minutes; default 60
    wellelev = site ground surface elevation in feet
    stickup = offset of measure point from ground in feet
    
    OUTPUT
    -----
    wellbarofinal = pandas dataframe with corrected water levels 
    
    This function uses pandas dataframes created using the 

    '''
    # resample data to make sample interval consistent
    baro = hourly_resample(barofile,0,sampint)
    well = hourly_resample(wellfile,0,sampint)
    # reassign `Level` to reduce ambiguity
    well['abs_feet_above_levelogger'] = well['Level']
    baro['abs_feet_above_barologger'] = baro['Level']
    # combine baro and well data for easy calculations, graphing, and manipulation
    wellbaro = pd.merge(well,baro,left_index=True,right_index=True,how='inner')
    wellbaro['adjusted_levelogger'] =  wellbaro['abs_feet_above_levelogger'] - wellbaro['abs_feet_above_barologger']
    
    breakpoints = []
    bracketedwls = {}

    for i in range(len(manualfile)+1):
        breakpoints.append(fcl(wellbaro, manualfile.index.to_datetime()[i-1]).name)

    last_man_wl,first_man_wl,last_tran_wl,driftlen = [],[],[],[]

    firstupper, firstlower, firstlev, lastupper, lastlower, lastlev = [],[],[],[],[],[]

    for i in range(len(manualfile)-1):
        # Break up time series into pieces based on timing of manual measurements
        bracketedwls[i+1] = wellbaro.loc[(wellbaro.index.to_datetime() > breakpoints[i+1])&(wellbaro.index.to_datetime() < breakpoints[i+2])]
        bracketedwls[i+1]['diff_wls'] = bracketedwls[i+1]['abs_feet_above_levelogger'].diff() 
        firstupper.append(np.mean(bracketedwls[i+1]['diff_wls'][2:41]) + 
                          np.std(bracketedwls[i+1]['diff_wls'][2:41])*2.0) # 2.2 std dev.
        firstlower.append(np.mean(bracketedwls[i+1]['diff_wls'][2:41]) - 
                          np.std(bracketedwls[i+1]['diff_wls'][2:41])*2.0) # 2.2 std dev.
        firstlev.append(bracketedwls[i+1].ix[1,'diff_wls']) # difference of first two values
        ## Examine Last Value
        lastupper.append(np.mean(bracketedwls[i+1]['diff_wls'][-41:-2]) + 
                         np.std(bracketedwls[i+1]['diff_wls'][-41:-2])*2.0) # 2.2 std dev.
        lastlower.append(np.mean(bracketedwls[i+1]['diff_wls'][-41:-2]) - 
                         np.std(bracketedwls[i+1]['diff_wls'][-41:-2])*2.0) # 2.2 std dev.
        lastlev.append(bracketedwls[i+1].ix[-1,'diff_wls']) # difference of last two values
        ## drop first value if 2.2 std dev beyond first 30 values
        if np.abs(firstlev[i]) > 0.1:
            if firstlev[i] > firstupper[i] or firstlev[i] < firstlower[i]:
                bracketedwls[i+1].drop(bracketedwls[i+1].index[0],inplace=True)
        ## drop last value if 2.2 std dev beyond last 30 values
        if np.abs(lastlev[i]) > 0.1:
            if lastlev[i] > lastupper[i] or lastlev[i] < lastlower[i]:
                bracketedwls[i+1].drop(bracketedwls[i+1].index[-1],inplace=True)

        bracketedwls[i+1].loc[:,'DeltaLevel'] = bracketedwls[i+1].loc[:,'adjusted_levelogger'] - bracketedwls[i+1].ix[0,'adjusted_levelogger']
        bracketedwls[i+1].loc[:,'MeasuredDTW'] = fcl(manualfile,breakpoints[i+1])[0] - bracketedwls[i+1].loc[:,'DeltaLevel']

        last_man_wl.append(fcl(manualfile,breakpoints[i+2])[0])
        first_man_wl.append(fcl(manualfile,breakpoints[i+1])[0])
        last_tran_wl.append(float(bracketedwls[i+1].loc[max(bracketedwls[i+1].index.to_datetime()),'MeasuredDTW']))
        driftlen.append(len(bracketedwls[i+1].index))
        bracketedwls[i+1].loc[:,'last_diff_int'] = np.round((last_tran_wl[i]-last_man_wl[i]),4)/np.round(driftlen[i]-1.0,4)
        bracketedwls[i+1].loc[:,'DriftCorrection'] = np.round(bracketedwls[i+1].loc[:,'last_diff_int'].cumsum()-bracketedwls[i+1].loc[:,'last_diff_int'],4)

    wellbarofixed = pd.concat(bracketedwls)
    wellbarofixed.reset_index(inplace=True)
    wellbarofixed.set_index('DateTime',inplace=True)
    # Get Depth to water below casing
    wellbarofixed.loc[:,'DTWBelowCasing'] = wellbarofixed['MeasuredDTW'] - wellbarofixed['DriftCorrection']

    # subtract casing height from depth to water below casing
    wellbarofixed.loc[:,'DTWBelowGroundSurface'] = wellbarofixed.loc[:,'DTWBelowCasing'] - stickup #well riser height

    # subtract depth to water below ground surface from well surface elevation
    wellbarofixed.loc[:,'WaterElevation'] = wellelev - wellbarofixed.loc[:,'DTWBelowGroundSurface']
    
    wellbarofinal = smoother(wellbarofixed, 'WaterElevation')
    
    return wellbarofinal
    
    
# clark's method
def clarks(df,bp,wl):
    '''
    Determines and plots barometric efficiency of a well using Clark's method

    INPUT
    ------
    df = pandas dataframe containing barometric pressure and water level data
    bp = barometric pressure column name in df
    wl = water level column name in df
    
    RETURNS
    ------
    m, b, r
    m = slope of regression line; barometric efficiency
    b = intercept of line fitting clark line
    r = r-squared value of line fitting clark line    
    '''
    
    df['dwl'] = df[wl].diff()
    df['dbp'] = df[bp].diff()
    
    df['beta'] = df['dbp']*df['dwl']
    df['Sbp'] = np.abs(df['dbp']).cumsum()
    df['Swl'] = df[['dwl','beta']].apply(lambda x: -1*np.abs(x[0]) if x[1]>0 else np.abs(x[0]), axis=1).cumsum()
    plt.figure()
    plt.plot(df['Sbp'],df['Swl'])
    regression = ols(y=df['Swl'], x=df['Sbp'])
    
    m = regression.beta.x
    b = regression.beta.intercept
    r = regression.r2
    
    y_reg = [df.ix[i,'Sbp']*m+b for i in range(len(df['Sbp']))]

    plt.plot(df['Sbp'],y_reg,
             label='Regression: Y = {m:.4f}X + {b:.5}\nr^2 = {r:.4f}\n BE = {be:.2f} '.format(m=m,b=b,r=r,be=m))
    plt.legend()
    plt.xlabel('Sum of Barometric Pressure Changes (ft)')
    plt.ylabel('Sum of Water-Level Changes (ft)')
    df.drop(['dwl','dbp','Sbp','Swl'],axis=1,inplace=True)
    return m,b,r


def Scat(df,bp,wl):
    '''
    Produces a scatter plot of changes in water level as a function of changes in barometric pressure

    INPUT
    ------
    df = pandas dataframe containing barometric pressure and water level data
    bp = barometric pressure column name in df
    wl = water level column name in df
    
    RETURNS
    ------
    m, b, r
    m = slope of regression line; barometric efficiency
    b = intercept of line fitting clark line
    r = r-squared value of line fitting clark line       
    '''
    df['dwl'] = df[wl].diff()
    df['dbp'] = df[bp].diff()

    regression = ols(y=df['dwl'], x=df['dbp'])
    m = regression.beta.x
    b = regression.beta.intercept
    r = regression.r2
    #r = (regression.beta.r)**2
    plt.scatter(y=df['dwl'], x=df['dbp'])

    y_reg = [df['dbp'][i]*m+b for i in range(len(df['dbp']))]

    plt.plot(df['dbp'],y_reg, 
             label='Regression: Y = {m:.4f}X + {b:.5}\nr^2 = {r:.4f}\n BE = {be:.2f} '.format(m=m,b=b,r=r,be=m))
    plt.legend()
    plt.xlabel('Sum of Barometric Pressure Changes (ft)')
    plt.ylabel('Sum of Water-Level Changes (ft)')
    return m,b,r
    
    
def baro_eff(df,bp,wl,lag=100,how='diff'):
    '''
    Apply the barometric response function technique outlined by Rasmussen and Crawford (1997). This method uses convolution to least squares regression to identify and remove barometric effects.
    
    INPUT
    ------
    df = pandas dataframe containing barometric pressure and water level data
    bp = barometric pressure column name in df
    wl = water level column name in df
    
    lag = number of samples to consider; default is 100
    how = method to use to standardize water level and barometric data; options are 'diff' (1st difference), 'mean' (value-mean), 'none'; default is diff    
    
    RETURNS
    ------
    negcumls = barometric response
    resid = corrected water level, with barometric effects removed
    lag_time = lag time of barometric response
    time = datetime of the data
    dwl = water level information used for calculations
    dbp = barometric pressure information used for calculations
    
    REFERENCE
    ------
    Rasmussen, T.C., and Crawford, L.A., 1997, Identifying and removing barometric pressure effects in confined and unconfined aquifers: Ground Water, v. 35, n. 3, pp. 502-511. doi: 10.1111/j.1745-6584.1997.tb00111.x
    '''
    df = df[[bp,wl]].dropna()

    if how=='diff':
        dwl = df[wl].diff().values[1:-1] #first differences
        dbp = df[bp].diff().values[1:-1] #first differences
    elif how=='mean':
        dwl = np.subtract(df[wl].values[1:-1],np.mean(df[wl].values[1:-1])) #values minus mean
        dbp = np.subtract(df[bp].values[1:-1],np.mean(df[bp].values[1:-1])) #values minus mean
    elif how=='norm':
        dwl = np.divide(np.subtract(df[wl].values[1:-1],np.mean(df[wl].values[1:-1])),np.std(df[wl].values[1:-1])) #values minus mean
        dbp = np.divide(np.subtract(df[bp].values[1:-1],np.mean(df[bp].values[1:-1])),np.std(df[bp].values[1:-1])) #values minus mean
    elif how=='none':
        dwl = df[wl].values[1:-1]
        dbp = df[dp].values[1:-1]
    else:
        dwl = df[wl].diff().values[1:-1] #first differences
        dbp = df[bp].diff().values[1:-1] #first differences
    
    # Calculate julian dates of data
    df['j_dates'] = df.index.to_julian_date()
    time = df.index.to_datetime()[1:-1]
    
    lag_time = df['j_dates'].diff().cumsum().values[1:-1]
    df.drop('j_dates',axis=1,in_place=True)
    # Calculate BP Response Function

    ## create lag matrix for regression
    bpmat = tools.lagmat(dbp, lag, original='in')
    ## transpose matrix to determine required length
    ## run least squared regression
    sqrd = np.linalg.lstsq(bpmat,dwl)
    wlls = sqrd[0]
    cumls = np.cumsum(wlls)
    negcumls = [-1*cumls[i] for i in range(len(cumls))]
    ymod = np.dot(bpmat,wlls)
    
    ## resid gives the residual of the bp
    resid=[(dwl[i] - ymod[i])+np.mean(df[wl].values[1:-1]) for i in range(len(dwl))]
    lag_trim = lag_time[0:len(cumls)]
    
    plt.figure()
    lag_trim = lag_time[0:len(negcumls)]
    plt.scatter(lag_trim*24,negcumls, label='b.p. alone')
    plt.xlabel('lag (hours)')
    plt.ylabel('barometric response')
    
    return negcumls, resid, lag_time, dwl, dbp, time