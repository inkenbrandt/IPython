-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
import re
import xmltodict
from datetime import datetime




def getfilename(path):
    # this function extracts the file name without file path or extension
    return path.split('\\').pop().split('/').pop().rsplit('.', 1)[0]


def compilation(inputfile):
    """
    This function reads multiple Solinst transducer files in a directory and generates a compiled Pandas dataframe.
    
    inputfile = complete file path to input files; use * for wildcard in file name
        example -> 'O:\\Snake Valley Water\\Transducer Data\\Raw_data_archive\\all\\LEV\\*baro*' picks any file containing 'baro'
    
    packages required:
        pandas as pd
        glob
        os
        xmltodict
    """
        
    # create empty dictionary to hold dataframes
    f={}

    # generate list of relevant files
    filelist = glob.glob(inputfile)

    # iterate through list of relevant files
    for infile in filelist:
        # get the extension of the input file
        filetype = os.path.splitext(infile)[1]
        # run computations using lev files
        if filetype=='.lev':
            # open text file
            with open(infile) as fd:
                # find beginning of data
                indices = fd.readlines().index('[Data]\n')

            # convert data to pandas dataframe starting at the indexed data line
            f[getfilename(infile)] = pd.read_table(infile, parse_dates=True, sep='     ', index_col=0,
                                           skiprows=indices+2, names=['DateTime','Level','Temperature'], skipfooter=1,engine='python')
            # add extension-free file name to dataframe
            f[getfilename(infile)]['name'] = getfilename(infile)
            
        # run computations using xle files
        elif filetype=='.xle':
            # open text file
            with open(infile) as fd:
                # parse xml
                obj = xmltodict.parse(fd.read(),encoding="ISO-8859-1")
            # navigate through xml to the data
            wellrawdata = obj['Body_xle']['Data']['Log']
            # convert xml data to pandas dataframe
            f[getfilename(infile)] = pd.DataFrame(wellrawdata)
            # get header names and apply to the pandas dataframe          
            f[getfilename(infile)][str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f[getfilename(infile)]['ch1']
            f[getfilename(infile)][str(obj['Body_xle']['Ch2_data_header']['Identification']).title()] = f[getfilename(infile)]['ch2']
  
            # add extension-free file name to dataframe
            f[getfilename(infile)]['name'] = getfilename(infile)
            # combine Date and Time fields into one field
            f[getfilename(infile)]['DateTime'] = pd.to_datetime(f[getfilename(infile)].apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
            f[getfilename(infile)] = f[getfilename(infile)].reset_index()
            f[getfilename(infile)] = f[getfilename(infile)].set_index('DateTime')
            f[getfilename(infile)] = f[getfilename(infile)].drop(['Date','Time','@id','ch1','ch2','index','ms'],axis=1)
        # run computations using csv files

        else:
            pass
    # concatonate all of the dataframes in dictionary f to one dataframe: g
    g = pd.concat(f)
    # remove multiindex and replace with index=Datetime
    g = g.reset_index()
    g = g.set_index(['DateTime'])
    # drop old indexes
    g = g.drop(['level_0'],axis=1)
    # remove duplicates based on index then sort by index
    g['ind']=g.index
    g.drop_duplicates(subset='ind',inplace=True)
    g.drop('ind',axis=1,inplace=True)
    g = g.sort()
    # ensure that the Level and Temperature data are in a float format
    g['Level'] = g['Level'].convert_objects(convert_numeric=True)
    g['Temperature'] = g['Temperature'].convert_objects(convert_numeric=True)
    outfile = g
    return outfile



def appendomatic(infile,existingfile):
    '''
    appends data from one table to an existing compilation
    this tool will delete and replace the existing file

    infile = input file
    existingfile = file you wish to append to
    '''

    # get the extension of the input file
    filetype = os.path.splitext(infile)[1]
    
    # run computations using lev files
    if filetype=='.lev':
        # open text file
        with open(infile) as fd:
            # find beginning of data
            indices = fd.readlines().index('[Data]\n')

        # convert data to pandas dataframe starting at the indexed data line
        f = pd.read_table(infile, parse_dates=True, sep='     ', index_col=0,
                                       skiprows=indices+2, names=['DateTime','Level','Temperature'], skipfooter=1,engine='python')
        # add extension-free file name to dataframe
        f['name'] = getfilename(infile)

    # run computations using xle files
    elif filetype=='.xle':
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
        unit = str(obj['Body_xle']['Ch1_data_header']['Unit']).lower()
        if unit == "feet":
            f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']
        elif unit == "kpa":
            f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']*0.33456
        elif unit == "mbar":
            f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']*0.0334552565551
        elif unit == "psi":
            f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']*2.306726
        else:
            f[str(obj['Body_xle']['Ch1_data_header']['Identification']).title()] = f['ch1']
            print "Unknown Units"
        # add extension-free file name to dataframe
        f['name'] = getfilename(infile)
        # combine Date and Time fields into one field
        f['DateTime'] = pd.to_datetime(f.apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
        f = f.reset_index()
        f = f.set_index('DateTime')
        f = f.drop(['Date','Time','@id','ch1','ch2','index','ms'],axis=1)
    
    # run computations using csv files
    elif filetype=='.csv':
        with open(infile) as fd:
        # find beginning of data
            try:
                indices = fd.readlines().index('Date,Time,ms,Level,Temperature\n')
            except ValueError:
                indices = fd.readlines().index(',Date,Time,100 ms,Level,Temperature\n')
        f = pd.read_csv(infile, skiprows=indices, skipfooter=1, engine='python')
        # add extension-free file name to dataframe
        f['name'] = getfilename(infile)
        # combine Date and Time fields into one field
        f['DateTime'] = pd.to_datetime(f.apply(lambda x: x['Date'] + ' ' + x['Time'], 1))
        f = f.reset_index()
        f = f.set_index('DateTime')
        f = f.drop(['Date','Time','ms','index'],axis=1)
            # skip other file types
    else:
        pass

    # ensure that the Level and Temperature data are in a float format
    f['Level'] = f['Level'].convert_objects(convert_numeric=True)
    f['Temperature'] = f['Temperature'].convert_objects(convert_numeric=True)
    h = pd.read_csv(existingfile,index_col=0,header=0,parse_dates=True)
    g = pd.concat([h,f])
    # remove duplicates based on index then sort by index
    g['ind']=g.index
    g.drop_duplicates(subset='ind',inplace=True)
    g.drop('ind',axis=1,inplace=True)
    g = g.sort()    
    os.remove(existingfile)
    g.to_csv(existingfile)


def make_files_table(folder, wellinfo):
    '''
    This tool will make a descriptive table (Pandas DataFrame) containing filename, date, and site id.
    For it to work properly, files must be named in the following fashion:
    siteid YYYY-MM-DD
    example: pw03a 2015-03-15.csv

    This tool assumes that if there are two files with the same name but different extensions, 
    then the datalogger for those data is a Solinst datalogger.

    folder = directory containing the newly downloaded transducer data
    '''

    filenames = next(os.walk(folder))[2]
    site_id, exten, dates, fullfilename = [],[],[],[]
    # parse filename into relevant pieces
    for i in filenames:
        site_id.append(i[:-15].lower())
        exten.append(i[-4:])
        dates.append(i[-14:-4])
        fullfilename.append(i)
    files = {'siteid':site_id,'extensions':exten,'date':dates,'full_file_name':fullfilename}
    files = pd.DataFrame(files)
    files['filedups'] = files.duplicated(subset='siteid')
    files['LoggerTypeID'] = files['filedups'].astype('int')+1
    files['LoggerTypeName']=files['LoggerTypeID'].apply(lambda x: 'Solinst' if x==2 else 'Global Water')
    files.drop_duplicates(subset='siteid',take_last=True,inplace=True)

    #wellinfo = pd.read_csv(wellinfofile,header=0,index_col=0)
    wellinfo = wellinfo[wellinfo['Well']!=np.nan]
    wellinfo["G_Elev_m"] = np.divide(wellinfo["GroundElevation"],3.2808)
    wellinfo['Well'] = wellinfo['Well'].apply(lambda x: str(x).lower().strip())
    files = pd.merge(files,wellinfo,left_on='siteid',right_on='Well')
    
    return files



def barodistance(wellinfo):
    '''
    Determines Closest Barometer to Each Well using wellinfo DataFrame
    '''
    barometers = {'barom':['pw03','pw10','pw19'], 'X':[240327.49,271127.67,305088.9], 
                  'Y':[4314993.95,4356071.98,4389630.71], 'Z':[1623.079737,1605.187759,1412.673738]}
    barolocal = pd.DataFrame(barometers)
    barolocal = barolocal.reset_index()
    barolocal.set_index('barom',inplace=True)

    wellinfo['pw03'] = np.sqrt((barolocal.loc['pw03','X']-wellinfo['UTMEasting'])**2 + 
                                   (barolocal.loc['pw03','Y']-wellinfo['UTMNorthing'])**2 + 
                                   (barolocal.loc['pw03','Z']-wellinfo['G_Elev_m'])**2)
    wellinfo['pw10'] = np.sqrt((barolocal.loc['pw10','X']-wellinfo['UTMEasting'])**2 + 
                                   (barolocal.loc['pw10','Y']-wellinfo['UTMNorthing'])**2 + 
                                   (barolocal.loc['pw10','Z']-wellinfo['G_Elev_m'])**2)
    wellinfo['pw19'] = np.sqrt((barolocal.loc['pw19','X']-wellinfo['UTMEasting'])**2 + 
                                   (barolocal.loc['pw19','Y']-wellinfo['UTMNorthing'])**2 + 
                                   (barolocal.loc['pw19','Z']-wellinfo['G_Elev_m'])**2)
    wellinfo['closest_baro'] = wellinfo[['pw03','pw10','pw19']].T.idxmin()
    return wellinfo



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
    if levelunit == "feet":
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

def getwellid(infile,wellinfo):
    m = re.search("\d", getfilename(infile))
    s = re.search("\s", getfilename(infile))
    if m.start() > 3:
        wellname = getfilename(infile)[0:m.start()].strip().lower()
    else:
        wellname = getfilename(infile)[0:s.start()].strip().lower()
    wellid = wellinfo[wellinfo['Well']==wellname]['WellID'].values[0]
    return wellname, wellid


def imp_new_well(infile, wellinfo, manual, baro):
    '''
    INPUT
    infile = full file path of well to import
    wellinfo = pandas dataframe containing infomation of snake valley wells
    manual = pandas dataframe containing manual water level measurements
    
    OUTPUT
    a pandas dataframe and a csv file
    
    This function imports xle (solinst) and csv (Global Water) transducer files, removes barometric pressure effects and corrects for drift.
    ''' 
    wellname, wellid = getwellid(infile,wellinfo) #see custom getwellid function
    if wellinfo[wellinfo['Well']==wellname]['LoggerTypeName'].values[0] == 'Solinst': # Reads Solinst Raw File
        f = new_xle_imp(infile)
        bse = f.index.to_datetime().minute[0]
        bp = wellinfo[wellinfo['Well']==wellname]['BE barologger']
        b = hourly_resample(baro[bp], bse)
        g = pd.merge(f,b,left_index=True,right_index=True,how='inner')
        
        g['MeasuredLevel'] = g['Level']         
        
        # Remove first and/or last measurements if the transducer was out of the water
        ## Examine First Value
        firstupper = np.mean(g['MeasuredLevel'].diff()[2:31]) + np.std(g['MeasuredLevel'].diff()[2:31])*2.2 # 2.2 std dev.
        firstlower = np.mean(g['MeasuredLevel'].diff()[2:31]) - np.std(g['MeasuredLevel'].diff()[2:31])*2.2 # 2.2 std dev.
        firstlev = g['MeasuredLevel'].diff()[1:2].values[0] # difference of first two values
        ## Examine Last Value
        lastupper = np.mean(g['MeasuredLevel'].diff()[-31:-2]) + np.std(g['MeasuredLevel'].diff()[-31:-2])*2.2 # 2.2 std dev.
        lastlower = np.mean(g['MeasuredLevel'].diff()[-31:-2]) - np.std(g['MeasuredLevel'].diff()[-31:-2])*2.2 # 2.2 std dev.
        lastlev = g['MeasuredLevel'].diff()[-2:-1].values[0] # difference of last two values
        ## drop first value if 2.2 std dev beyond first 30 values
        if np.abs(firstlev) > 0.1:
            if firstlev > firstupper or firstlev < firstlower:
                g.drop(g.index[0],inplace=True)
        ## drop last value if 2.2 std dev beyond last 30 values
        if np.abs(lastlev) > 0.1:
            if lastlev > lastupper or lastlev < lastlower:
                g.drop(g.index[-1],inplace=True)
        
        glist = f.columns.tolist()
        if 'Temperature' in glist:
            g['Temp'] = g['Temperature']
            g.drop(['Temperature'],inplace=True,axis=1)
        elif 'Temp' in glist:
            pass
        # Get Baro Efficiency
        be = wellinfo[wellinfo['WellID']==wellid]['BaroEfficiency']
        be = be.iloc[0]
    
        # Barometric Efficiency Correction
        g['BaroEfficiencyCorrected'] = g['MeasuredLevel'] - g[wellinfo[wellinfo['Well']==wellname]['BE barologger']] + be*g[wellinfo[wellinfo['Well']==wellname]['BE barologger']]
    else: # Reads Global Water Raw File
        f = pd.read_csv(infile,skiprows=1,parse_dates=[[0,1]])
        f = f.reset_index()
        f['DateTime'] = f['Date_ Time']
        f['Level'] = f[' Feet']
        flist = f.columns.tolist()
        if ' Temp C' in flist:
            f['Temperature'] = f[' Temp C']
            f['Temp'] = f['Temperature']
            f.drop([' Temp C','Temperature'],inplace=True,axis=1)
        else:
            f['Temp'] = np.nan
        f = f.set_index('DateTime')
        f['date'] = f.index.to_julian_date().values
        f['datediff'] = f['date'].diff()
        f = f[f['datediff']>0]
        f = f[f['datediff']<1]
        f = f.resample("60Min")
        f = f.interpolate(method='time')
        f.drop(['index',u' Volts',' Feet',u'date',u'datediff'],inplace=True,axis=1)        
        bse = f.index.to_datetime().minute[0]
        bp = wellinfo[wellinfo['Well']==wellname]['BE barologger']
        b = hourly_resample(baro[bp], bse)
        g = pd.merge(f,b,left_index=True,right_index=True,how='inner')
        g['MeasuredLevel'] = g['Level']
        
        # Remove first and/or last measurements if the transducer was out of the water
        ## Examine First Value
        firstupper = np.mean(g['MeasuredLevel'].diff()[2:31]) + np.std(g['MeasuredLevel'].diff()[2:31])*2.2 # 2.2 std dev.
        firstlower = np.mean(g['MeasuredLevel'].diff()[2:31]) - np.std(g['MeasuredLevel'].diff()[2:31])*2.2 # 2.2 std dev.
        firstlev = g['MeasuredLevel'].diff()[0:1].values[0] # difference of first two values
        ## Examine Last Value
        lastupper = np.mean(g['MeasuredLevel'].diff()[-31:-2]) + np.std(g['MeasuredLevel'].diff()[-31:-2])*2.2 # 2.2 std dev.
        lastlower = np.mean(g['MeasuredLevel'].diff()[-31:-2]) - np.std(g['MeasuredLevel'].diff()[-31:-2])*2.2 # 2.2 std dev.
        lastlev = g['MeasuredLevel'].diff()[-2:-1].values[0] # difference of last two values
        ## drop first value if 2.2 std dev beyond first 30 values
        if firstlev > 0.1:
            if firstlev > firstupper or firstlev < firstlower:
                g.drop(g.index[0],inplace=True)
        ## drop last value if 2.2 std dev beyond last 30 values
        if lastlev > 0.1:
            if lastlev > lastupper or lastlev < lastlower:
                g.drop(g.index[-1],inplace=True)
        
        # Get Baro Efficiency
        be = wellinfo[wellinfo['WellID']==wellid]['BaroEfficiency']
        be = be.iloc[0]
    
        # Barometric Efficiency Correction
        g['BaroEfficiencyCorrected'] = g['MeasuredLevel'] + be*g[wellinfo[wellinfo['Well']==wellname]['BE barologger']]
                
    g['DeltaLevel'] = g['BaroEfficiencyCorrected'] - g['BaroEfficiencyCorrected'][0]
    
    # Match manual water level to closest date
    g['MeasuredDTW'] = fcl(manual[manual['WellID']== wellid],min(g.index.to_datetime()))[1]-g['DeltaLevel']

    # Drift Correction
    #lastdtw = g['MeasuredDTW'][-1]
    last = fcl(manual[manual['WellID']== wellid],max(g.index.to_datetime()))[1]
    first = fcl(manual[manual['WellID']== wellid],min(g.index.to_datetime()))[1]
    lastg = float(g[g.index.to_datetime()==max(g.index.to_datetime())]['MeasuredDTW'].values)
    driftlen = len(g.index)
    g['last_diff_int'] = np.round((lastg-last),4)/np.round(driftlen-1.0,4)
    g['DriftCorrection'] = np.round(g['last_diff_int'].cumsum()-g['last_diff_int'],4)
    
    # Assign well id to column
    g['WellID'] = wellid
    
    # Get Depth to water below casing
    g['DTWBelowCasing'] = g['MeasuredDTW'] - g['DriftCorrection']

    # subtract casing height from depth to water below casing
    g['DTWBelowGroundSurface'] = g['DTWBelowCasing'] - wellinfo[wellinfo['WellID']==wellid]['Offset'].values[0]
    
    # subtract depth to water below ground surface from well surface elevation
    g['WaterElevation'] = wellinfo[wellinfo['WellID']==wellid]['GroundElevation'].values[0] - g['DTWBelowGroundSurface']
    
    # assign tape value
    g['Tape'] = 0
    g['MeasuredBy'] = ''
    
    # generate new file
    pathlist = os.path.splitext(infile)[0].split('\\')
    outpath = pathlist[0] + '\\' + pathlist[1] + '\\' + pathlist[2] + '\\' + pathlist[3] + '\\' + pathlist[4] + '\\' + str(wellname) + '.csv'  
    g['DateTime'] = g.index.to_datetime()
    g.to_csv(outpath, index=False, columns= ["WellID","DateTime","MeasuredLevel","Temp","BaroEfficiencyCorrected","DeltaLevel",
                                             "MeasuredDTW","DriftCorrection","DTWBelowCasing","DTWBelowGroundSurface",
                                             "WaterElevation","Tape","MeasuredBy"])
    return g


def hourly_resample(df,bse=0):
    df = df.resample('1Min')
    df = df.interpolate(method='time')
    df = df.resample('60Min', how='first',closed='left',label='left', base=bse)
    return df



def fcl(df, dtObj):
    '''
    finds closest date index in a dataframe to a date object
    
    df = dataframe
    dtObj = date object
    
    taken from: http://stackoverflow.com/questions/15115547/find-closest-row-of-dataframe-to-given-time-in-pandas
    '''
    return df.iloc[np.argmin(np.abs(df.index.to_pydatetime() - dtObj))]


