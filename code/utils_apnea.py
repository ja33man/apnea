# Import of libraries for working with the data
import pandas as pd
import os
import wfdb
import numpy as np

#-----------------------------------------------------------------------------------------------

# Additional libraries and settings for plotting 
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
from scipy.signal import resample
from sklearn.metrics import confusion_matrix, classification_report
#-----------------------------------------------------------------------------------------------

# Fixing imbalanced classes
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
ROS = RandomOverSampler(sampling_strategy='all', return_indices=True, random_state=1)
RUS = RandomUnderSampler(sampling_strategy='auto', return_indices=True, random_state=10)

#-----------------------------------------------------------------------------------------------

# Keras libraries
from keras.models import Model, load_model
from keras.utils import plot_model
import tensorflow as tf
from keras.engine import Layer, InputSpec
from keras import initializers, constraints, models, optimizers, regularizers, layers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
import keras
from keras.utils.vis_utils import plot_model
from keras.initializers import glorot_uniform
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

#-----------------------------------------------------------------------------------------------

#Setting up fonts for plotting
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 21
plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Plotting parameters for better view
rcParams['figure.figsize'] = 36, 6
# Only used if Jupyter Theme is Dark
#rcParams['xtick.color'] = 'white'
#rcParams['ytick.color'] = 'white'

#-----------------------------------------------------------------------------------------------


# Apnea / RERA dictionary
rera_dict = {'(arousal_bruxism':2,
 '(arousal_noise':2,
 '(arousal_plm':2,
 '(arousal_rera':2,
 '(arousal_snore':2,
 '(arousal_spontaneous':2,
 '(resp_centralapnea':1,
 '(resp_cheynestokesbreath':1,
 '(resp_hypopnea':1,
 '(resp_hypoventilation':1,
 '(resp_mixedapnea':1,
 '(resp_obstructiveapnea':1,
 '(resp_partialobstructive':1,
 'N1':None,
 'N2':None,
 'N3':None,
 'R':None,
 'W':None,
  'arousal_bruxism)':0,
 'arousal_noise)':0,
 'arousal_plm)':0,
 'arousal_rera)':0,
 'arousal_snore)':0,
 'arousal_spontaneous)':0,
 'resp_centralapnea)':0,
 'resp_cheynestokesbreath)':0,
 'resp_hypopnea)':0,
 'resp_hypoventilation)':0,
 'resp_mixedapnea)':0,
 'resp_obstructiveapnea)':0,
 'resp_partialobstructive)':0}

# Sleep stage dictionary
stage_dict = {
 'N1':1,
 'N2':2,
 'N3':3,
 'R':4,
 'W':0}
#-----------------------------------------------------------------------------------------------

# Path to folder with downloaded data for PhysioNet/CinC Challenge 2018 (https://physionet.org/physiobank/database/challenge/2018/)
# Training data can be downloaded using torrent or using wget  -r -np -nH --cut-dirs=4 -R "index.html*" -e robots=off "https://physionet.org/physiobank/database/challenge/2018/training/"
pathToDataDir = os.path.abspath('../../physio_data/training/')

# Getting all training data records
dfCatalog = pd.read_csv((pathToDataDir+'/RECORDS'), header=None, names=['Folder'])

# Challenge name/location for wfdb package (https://github.com/MIT-LCP/wfdb-python), if data will be pulled from the web directly
# removing tr03-0394, tr07-0016 and tr07-0023 no labels for apnea events
challenge_name = 'challenge/2018/training/' 
dfCatalog.drop([40], inplace=True)
dfCatalog.drop([513], inplace=True)
dfCatalog.drop([514], inplace=True)
dfCatalog.reset_index(inplace=True)
dfCatalog.drop(['index'],axis=1,inplace=True)

# Available channels/signals
chan_dict = {'F3-M2':0, 'F4-M1':1, 'C3-M2':2, 'C4-M1':3, 'O1-M2':4,
             'O2-M1':5, 'E1-M2':6, 'Chin1-Chin2':7, 'ABD':8, 'CHEST':9, 
             'AIRFLOW':10, 'SaO2':11, 'ECG':12}

# 3 signals will be used to train neural network
channels=[9,8,11] 

#-----------------------------------------------------------------------------------------------

# Data and Labels Generator
def generateDataPerPatient(patientID, channels = channels, online = True, apnea = True):
    """
    Function
    --------
    generateDataPerPatient

    Generate, clean and combine data and labels for each patient in PhysioNet Challenge 2018 database

    Parameters
    ----------
    patientID : int
        from dfCatalog 
        
    channels : list
        if only certain signals are needed ('F3-M2':0, 'F4-M1':1, 'C3-M2':2, 'C4-M1':3, 'O1-M2':4,
                                                         'O2-M1':5, 'E1-M2':6, 'Chin1-Chin2':7, 'ABD':8, 'CHEST':9, 
                                                         'AIRFLOW':10, 'SaO2':11, 'ECG':12) 
                                                         
    online : bool
        if True data is pulled from the web if False data is pulled from local pathToDataDir folder
        
    apnea  : bool
        if True labels outputs for Apnea/RERA events if False labels outputs for Sleep Stage categories
   
    Returns
    -------
    df.values : array
        numpy array with signals and labels
        
    fields['n_sig'] : int
        number of signals

    Example
    -------
    >>> trainData, numberOfChannels = generateTestDataPerPatient(patientID = 11, channels = [0,3,5], online = True, apnea = True)
    """
    if online:
        # Should be used if labels pulled from the web
        labels = wfdb.rdann(os.path.split(dfCatalog['Folder'].iloc[patientID])[0], pb_dir='{0}/{1}/'.format(challenge_name ,os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), extension='arousal')  
        
        # Should be used if signals pulled from the web
        signals, fields = wfdb.rdsamp(os.path.split(dfCatalog['Folder'].iloc[patientID])[0],pb_dir='{0}/{1}/'.format(challenge_name ,os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), channels=channels) 
        
    else:
        # Should be used if labels stored locally
        labels = wfdb.rdann(pathToDataDir+'/{0}/{0}'.format(os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), extension='arousal')
        
        # Should be used if signals stored locally
        signals, fields = wfdb.rdsamp(pathToDataDir+'/{0}/{0}'.format(os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), channels=channels) 
    #--------------------------------------------------------------------------------------------
    
    if apnea:
        # Combining apnea/RERA labels with signals for binary or multiclass classification
        # Loading Labels into DataFrame
        dfL = pd.DataFrame(labels.aux_note, index=labels.sample, columns=['all_events'])
        
        # Mapping apnea/RERA events data
        dfL['apnea_rera']  = dfL['all_events'].map(rera_dict)
        
        # Filling NaN data
        dfL['apnea_rera'].fillna(method = 'ffill', inplace=True)
        dfL['apnea_rera'].fillna(value = 0, inplace=True)
        
        # Converting to one-hot encoded labels
        dfLabelsCat = pd.DataFrame(columns=range(3))
        dfLabelsCat = dfLabelsCat.append(pd.get_dummies(dfL['apnea_rera']))
        
        # Adding index 0 for correct fillna when labels will be merged with signals
        dfLabelsCat.loc[0] = 0
        dfLabelsCat = dfLabelsCat.sort_index(ascending=True)
        dfLabelsCat.fillna(value=0, inplace=True)
        
        # Combining signals with labels and filling gaps
        df = pd.DataFrame(data=signals, columns=fields['sig_name'])
        df = df.join(dfLabelsCat)
        df.fillna(method='ffill', inplace=True)
        df.fillna(value=0, inplace=True)
        df.drop(columns=0, inplace=True)
        df = df[df[2] != 1]
        
    else:
        # Combining sleep stage labels with signals for multiclass classification
        # Loading Labels into DataFrame
        dfL = pd.DataFrame(labels.aux_note, index=labels.sample, columns=['all_events'])
        
        # Mapping sleep stage events data
        dfL['sleep_stage']  = dfL['all_events'].map(stage_dict)
        
        # Filling NaN data
        dfL['sleep_stage'].fillna(method = 'ffill', inplace=True)
        dfL['sleep_stage'].fillna(value = 0, inplace=True)
        
        # Converting to one-hot encoded labels
        dfLabelsCat = pd.DataFrame(columns=range(5))
        dfLabelsCat = dfLabelsCat.append(pd.get_dummies(dfL['sleep_stage']))
        
        # Adding index 0 for correct fillna when labels will be merged with signals
        dfLabelsCat.loc[0] = 0
        dfLabelsCat = dfLabelsCat.sort_index(ascending=True)
        dfLabelsCat.fillna(value=0, inplace=True)
        
        # Combining signals with labels and filling gaps
        df = pd.DataFrame(data=signals, columns=fields['sig_name'])
        df = df.join(dfLabelsCat)
        df.fillna(method='ffill', inplace=True)
        df.fillna(value=0, inplace=True)
        df.drop(columns=0, inplace=True)
        
    return df.values, fields['n_sig']

#-----------------------------------------------------------------------------------------------

# For binary classification apnea/rera or normal
def generate_arrays_from_file_withoutBatch(trainData, numberOfChannels, sampleShift, bufferSizeInSec, samplingRateInHz, upsample = True, binary = True, newFreq = True, newSamplingRateInHz = None):
    """
    Function
    --------
    generate_arrays_from_file_withoutBatch

    Splits continuous whole night data into short intervals for classification, and shifts splits for data augmentation

    Parameters
    ----------
    trainData : array
        Numpy array of continuous signals and labels
    
    numberOfChannels : int
        Total number of signal channels present in the data
        
    sampleShift : int
        Shift of data in seconds for data augmentation
    
    bufferSizeInSec : int
        Size of intervals for classification task in seconds
    
    samplingRateInHz : int
        Input sampling rate in Hz

    upsample : bool
        Over or undersample data to fix imbalanced dataset
    
    binary : bool
        Binary or multiclass classification
        
    newFreq : bool
        if resampling is needed
    
    newSamplingRateInHz : int
        New sampling rate in seconds
    
    Returns
    -------
    x : array
        Numpy array of signals data
    
    y : array
        1D array for binary classification: 0 - Normal interval, 1 - Apnea event, or array of one-hot encoded labels
    
    Example
    -------
    >>> xTrain, yTrain = generate_arrays_from_file_withoutBatch(trainData, numberOfChannels = 3, sampleShift = 12, bufferSizeInSec = 60, samplingRateInHz = 200, upsample = True, binary = True, newFreq = True, newSamplingRateInHz = 100)
    """
    #Shifting dataset by sampleShift in seconds
    trainData = trainData[sampleShift:]
    
    # Dividing data into intervals of a size bufferSizeInSec
    batch = np.array(trainData[:((trainData.shape[0]//(samplingRateInHz*bufferSizeInSec))*(samplingRateInHz*bufferSizeInSec))]).reshape((-1,samplingRateInHz*bufferSizeInSec,trainData.shape[1]))
    
    # Spliting data into signals and labels    
    x = np.nan_to_num(batch[:,:,:numberOfChannels])
    y = np.nan_to_num((batch[:,:,numberOfChannels:]).max(axis=1))
    
    # Deleting intervals where SpO2  level dropped signifficantly (below 50 %), likely due to equipment disconnect
    y = np.delete(y, list(set(np.nonzero((x[:,:,-1]<50))[0])), axis=0)
    x = np.delete(x, list(set(np.nonzero((x[:,:,-1]<50))[0])), axis=0)
    
    # Normalization of the signals between 0 and 1
    for j in range(x.shape[0]):
        x[j,:,0] = (x[j,:,0] - x[j,:,0].min(axis=0)) /np.nanmax(np.ptp(x[j,:,0], axis=0))
        x[j,:,1] = (x[j,:,1] - x[j,:,1].min(axis=0)) /np.nanmax(np.ptp(x[j,:,1], axis=0))
        x[j,:,2] /= 100
        
        
    #y = np.delete(y, list(set(np.nonzero((x[:,:,-1]<0.5))[0])), axis=0)
    #x = np.delete(x, list(set(np.nonzero((x[:,:,-1]<0.5))[0])), axis=0)
    
    # Checking for NaN
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    
    # Additional cleanup for very large small numbers
    for ch in range(numberOfChannels):
        y = np.delete(y, list(set(np.nonzero((x[:,:,ch]==np.nan))[0])), axis=0)
        x = np.delete(x, list(set(np.nonzero((x[:,:,ch]==np.nan))[0])), axis=0)
        y = np.delete(y, list(set(np.nonzero((x[:,:,ch]==np.inf))[0])), axis=0)
        x = np.delete(x, list(set(np.nonzero((x[:,:,ch]==np.inf))[0])), axis=0)
        
    # Binary or Multiclass classification
    if binary:
        y = y[:,:].max(axis=1)
    else:
        y = y[:,:]
        
    # removing 0 signal
    ind_0 = list(set(np.where(x != 0)[0])) 
    x = x[ind_0]
    y = y[ind_0]
    
    # Fixing imbalanced data
    if upsample:
        _, _, ind = ROS.fit_sample(x[:,:,0], y[:])
    else:
        _, _, ind = RUS.fit_sample(x[:,:,0], y[:])
        
    # Shuffling data
    np.random.shuffle(ind)
    x = x[ind]
    y = y[ind]
    
    # Resampling to a new sampling frequency
    if newFreq:
        x = resample(x, int(bufferSizeInSec*newSamplingRateInHz), axis=1)
    
    return x,y

# Data and Labels Generator for Test 

def generateTestDataPerPatient(patientID, channels = channels, online = True):
    """
    Function
    --------
    generateTestDataPerPatient

    Generate, clean and combine data and labels for each patient in PhysioNet Challenge 2018 database

    Parameters
    ----------
    patientID : int
        from dfCatalog 
        
    channels : list
        if only certain signals are needed ('F3-M2':0, 'F4-M1':1, 'C3-M2':2, 'C4-M1':3, 'O1-M2':4,
                                                         'O2-M1':5, 'E1-M2':6, 'Chin1-Chin2':7, 'ABD':8, 'CHEST':9, 
                                                         'AIRFLOW':10, 'SaO2':11, 'ECG':12) 
                                                         
    online : bool
        if True data is pulled from the web if False data is pulled from local pathToDataDir folder
   
    Returns
    -------
    df.values : array
        numpy array with signals and labels
        
    fields['n_sig'] : int
        number of signals

    Example
    -------
    >>> testData, numberOfChannels = generateTestDataPerPatient(patientID = 11, channels = [0,3,5], online = True)
    """

    if online:
        # Should be used if labels pulled from the web
        labels = wfdb.rdann(os.path.split(dfCatalog['Folder'].iloc[patientID])[0], pb_dir='{0}/{1}/'.format(challenge_name ,os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), extension='arousal')  
        
        # Should be used if signals pulled from the web
        signals, fields = wfdb.rdsamp(os.path.split(dfCatalog['Folder'].iloc[patientID])[0],pb_dir='{0}/{1}/'.format(challenge_name ,os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), channels=channels) 
        
    else:
        # Should be used if labels stored locally
        labels = wfdb.rdann(pathToDataDir+'/{0}/{0}'.format(os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), extension='arousal')
        
        # Should be used if signals stored locally
        signals, fields = wfdb.rdsamp(pathToDataDir+'/{0}/{0}'.format(os.path.split(dfCatalog['Folder'].iloc[patientID])[0]), channels=channels) 
        
        
    #--------------------------------------------------------------------------------------------
    # Combining apnea labels with signals for binary classification
    # Loading Labels into DataFrame
    dfL = pd.DataFrame(labels.aux_note, index=labels.sample, columns=['all_events'])
    
    # Mapping apnea events data
    dfL['apnea_rera']  = dfL['all_events'].map(rera_dict)
    
    # Filling NaN data
    dfL['apnea_rera'].fillna(method = 'ffill', inplace=True)
    dfL['apnea_rera'].fillna(value = 0, inplace=True)
    
    # Converting to one-hot encoded labels
    dfLabelsCat = pd.DataFrame(columns=range(3))
    dfLabelsCat = dfLabelsCat.append(pd.get_dummies(dfL['apnea_rera']))
    
    # Adding index 0 for correct fillna when labels will be merged with signals
    dfLabelsCat.loc[0] = 0
    dfLabelsCat = dfLabelsCat.sort_index(ascending=True)
    dfLabelsCat.fillna(value=0, inplace=True)
    
    # Combining signals with labels and filling gaps
    df = pd.DataFrame(data=signals, columns=fields['sig_name'])
    df = df.join(dfLabelsCat)
    df.fillna(method='ffill', inplace=True)
    df.fillna(value=0, inplace=True)
    df.drop(columns=0, inplace=True)
    df = df[df[2] != 1]
    return df.values, fields['n_sig']

#-----------------------------------------------------------------------------------------------

# For binary classification apnea or normal generate intervals

def generateBatch(testData, numberOfChannels, bufferSizeInSec, samplingRateInHz, newFreq = True, newSamplingRateInHz = 100):
    """
    Function
    --------
    generateBatch

    Splits continuous whole night data into short intervals for classification

    Parameters
    ----------
    testData : array
        Numpy array of continuous signals and labels
    
    numberOfChannels : int
        Total number of signal channels present in the data
    
    bufferSizeInSec : int
        Size of intervals for classification task in seconds
    
    samplingRateInHz : int
        Input sampling rate in Hz
    
    newFreq : bool
        if resampling is needed
    
    newSamplingRateInHz : int
        New sampling rate in seconds
    
    Returns
    -------
    x : array
        Numpy array of signals data
    
    y : array
        1D array for binary classification: 0 - Normal interval, 1 - Apnea event
    
    y2 : array
        2D array that shows interval of Apnea event

    Example
    -------
    >>> xTest, yTest, y2Test = generateBatch(testData, numberOfChannels = 3, bufferSizeInSec = 60, samplingRateInHz = 200, newFreq = True, newSamplingRateInHz = 100)
    """
    
    # Dividing data into intervals of a size bufferSizeInSec
    batch = np.array(testData[:((testData.shape[0]//(samplingRateInHz*bufferSizeInSec))*(samplingRateInHz*bufferSizeInSec))]).reshape((-1,samplingRateInHz*bufferSizeInSec,testData.shape[1]))
    
    # Spliting data into signals and labels
    x = np.nan_to_num(batch[:,:,:numberOfChannels])
    y2 = np.nan_to_num((batch[:,:,numberOfChannels:]))
    y = np.nan_to_num((batch[:,:,numberOfChannels:]).max(axis=1))
    
    # Deleting intervals where SpO2  level dropped signifficantly (below 50 %), likely due to equipment disconnect
    y = np.delete(y, list(set(np.nonzero((x[:,:,-1]<50))[0])), axis=0)
    y2 = np.delete(y2, list(set(np.nonzero((x[:,:,-1]<50))[0])), axis=0)
    x = np.delete(x, list(set(np.nonzero((x[:,:,-1]<50))[0])), axis=0)
    
    # Normalization of the signals between 0 and 1
    for j in range(x.shape[0]):
        x[j,:,0] = (x[j,:,0] - x[j,:,0].min(axis=0)) /np.nanmax(np.ptp(x[j,:,0], axis=0))
        x[j,:,1] = (x[j,:,1] - x[j,:,1].min(axis=0)) /np.nanmax(np.ptp(x[j,:,1], axis=0))
        x[j,:,2] /= 100
        
    
    #y = np.delete(y, list(set(np.nonzero((x[:,:,-1]<0.5))[0])), axis=0)
    #y2 = np.delete(y2, list(set(np.nonzero((x[:,:,-1]<0.5))[0])), axis=0)
    #x = np.delete(x, list(set(np.nonzero((x[:,:,-1]<0.5))[0])), axis=0)
    
    # Check for NaN
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    
    # Additional cleanup for very large small numbers
    for ch in range(numberOfChannels):
        y = np.delete(y, list(set(np.nonzero((x[:,:,ch]==np.nan))[0])), axis=0)
        y2 = np.delete(y2, list(set(np.nonzero((x[:,:,ch]==np.nan))[0])), axis=0)
        x = np.delete(x, list(set(np.nonzero((x[:,:,ch]==np.nan))[0])), axis=0)
        y = np.delete(y, list(set(np.nonzero((x[:,:,ch]==np.inf))[0])), axis=0)
        y2 = np.delete(y2, list(set(np.nonzero((x[:,:,ch]==np.inf))[0])), axis=0)
        x = np.delete(x, list(set(np.nonzero((x[:,:,ch]==np.inf))[0])), axis=0) 
        
    # Combining labels: one label per interval
    y = y[:,:].max(axis=1)
    
    # Resampling to a new sampling frequency
    if newFreq:
        x = resample(x, int(bufferSizeInSec*newSamplingRateInHz), axis=1)
        y2 = resample(y2, int(bufferSizeInSec*newSamplingRateInHz), axis=1)
        
    return x, y, y2


#-----------------------------------------------------------------------------------------------

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

#-----------------------------------------------------------------------------------------------

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = layers.SeparableConv1D(filters = F1, kernel_size = (1), strides = (1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2a')(X)
    X = layers.Activation('swish')(X)
    
    # Second component of main path
    X = layers.SeparableConv1D(filters = F2, kernel_size = (f), strides = (1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path 
    X = layers.SeparableConv1D(filters = F3, kernel_size = (1), strides = (1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2c')(X)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Add()([X_shortcut, X])
    X = layers.Activation('relu')(X)
    
    return X

# convolutional_block

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = layers.Conv1D(F1, (1), strides = (s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2a')(X)
    X = layers.Activation('swish')(X)

    # Second component of main path 
    X = layers.Conv1D(filters = F2, kernel_size = (f), strides = (1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2b')(X)
    X = layers.Activation('relu')(X)

    # Third component of main path
    X = layers.Conv1D(filters = F3, kernel_size = (1), strides = (1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = layers.BatchNormalization(axis = -1, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = layers.Conv1D(filters = F3, kernel_size = (1), strides = (s), padding = 'valid', name = conv_name_base + '1',
               kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = layers.BatchNormalization(axis = -1, name = bn_name_base + '1')(X_shortcut)
    X = layers.Add()([X_shortcut, X])
    X = layers.Activation('relu')(X)

    return X

#-----------------------------------------------------------------------------------------------
from sklearn.metrics import coverage_error, label_ranking_average_precision_score, accuracy_score, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

#-----------------------------------------------------------------------------------------------

