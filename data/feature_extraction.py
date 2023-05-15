import numpy as np
import scipy as sp
from scipy import stats
from scipy import signal
from numpy.lib.format import open_memmap

train_windows = np.load("point-25-sec/train_windows.npy", mmap_mode='r')
test_windows = np.load("point-25-sec/test_windows.npy", mmap_mode='r')

train_features = open_memmap('point-25-sec/train_features.npy', mode='w+', dtype=np.float32, shape=(train_windows.shape[0], 161))
test_features = open_memmap('point-25-sec/test_features.npy', mode='w+', dtype=np.float32, shape=(test_windows.shape[0], 161))

fs = 12 # ~0.25s window size

def corr_grav(x,y):
    corr = np.corrcoef(x,y)[0][1]
    if np.isnan(corr):
          corr = 0
    return corr

def cv_gravity(data):
        cv = stats.variation(data)
        if np.isnan(cv):
          cv = 0  # some might return NaN bc mean and std are 0
        return cv

def freq_cent(fft_ampl, fft_freq):
        sums = np.sum(fft_ampl)
        sums = np.where(sums, sums, 1.)  # Avoid dividing by zero
        return np.sum(fft_freq * fft_ampl) / sums

# GENERATE TRAIN FEATURES
i = 0 # counter
for window in train_windows:
    window_features = []
    # print(f"window shape: {window.shape}")

    b_mag = (np.linalg.norm(window[:3,:],axis=0))
    t_mag = (np.linalg.norm(window[3:,:],axis=0))
    # print(f"b_mag shape: {b_mag.shape}")

    b_mag = b_mag.reshape(list(b_mag.shape)+[1])
    t_mag = t_mag.reshape(list(t_mag.shape)+[1])
    # print(f"b_mag reshape: {b_mag.shape}")

    window = np.transpose(window)
    # print(f"window transpose: {window.shape}")

    window_withMag = np.concatenate([window,b_mag,t_mag], axis=1)
    # print(f"window_withMag shape: {window_withMag.shape}")
    
    window_withMag = np.transpose(window_withMag)
    # print(f"window_withMag transpose: {window_withMag.shape}")

    #  Filtering
    # create the LP filter
    # 1Hz 4th order LP Butterworth filter
    b, a  = signal.butter(4, Wn=1, fs=fs, btype='lowpass')

    # create the HP filter
    # 1Hz 4th order HP Butterworth filter
    d, c = signal.butter(4, Wn=1, fs=fs, btype='highpass')

    window_grav = signal.filtfilt(b, a, window_withMag, axis=1, padlen=10)
    window_mot = signal.filtfilt(d, c, window_withMag, axis=1, padlen=10)

    #print(len(window_features))
    # kept these as .append instead of += bc when I tested it, it threw an error when including the means later on
    window_features.append(corr_grav(window_withMag[0],window_withMag[1]))
    #print(corr_grav(window_withMag[0],window_withMag[1]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[2]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[3]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[1],window_withMag[2]))
    window_features.append(corr_grav(window_withMag[1],window_withMag[3]))
    window_features.append(corr_grav(window_withMag[1],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[1],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[2],window_withMag[3]))
    window_features.append(corr_grav(window_withMag[2],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[2],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[3],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[3],window_withMag[5]))
    window_features.append(corr_grav(window_withMag[4],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[6],window_withMag[7]))

    #print(len(window_features))


    # GRAVITY:
    #  Mean over all axes
    #print(np.concatenate(window_grav[0]).mean())
    window_features.append(np.mean(window_grav))
    #print(len(window_features))

    # Mean
    window_features+=[np.mean(x) for x in window_grav]

    # Median
    window_features+=[np.median(x) for x in window_grav]

    # Standard Deviation
    window_features+=[np.std(x) for x in window_grav]

    #  Coefficient of Variation
    window_features+=[cv_gravity(x) for x in window_grav]
    
    #  Min Value
    window_features+=[np.min(x) for x in window_grav]

    # 25th Percentile
    window_features+=[np.percentile(x,25) for x in window_grav]

    # 50th Percentile
    window_features+=[np.percentile(x,50) for x in window_grav]

    # 75th Percentile
    window_features+=[np.percentile(x,75) for x in window_grav]

    # Max Value
    window_features+=[np.max(x) for x in window_grav]

    # ACCELERATION:
    # Skew
    window_features+=[stats.skew(x) for x in window_mot]

    # Kurtosis
    window_features+=[stats.kurtosis(x) for x in window_mot]

    # Signal Energy
    window_features+=[np.sum(x**2) for x in window_mot]

    # Frequency-Domain Mean
    # FFT
    window_fft = np.abs(np.fft.fft(window_withMag, axis=0))

    window_features+=[np.mean(x) for x in window_fft]

    # Frequency-Domain Standard Deviation
    window_features+=[np.std(x) for x in window_fft]

    # Dominant Frequency
    window_features+=[np.fft.fftfreq(window.shape[0])[x.argmax(axis=0)] for x in window_fft]

    # Dominant Frequency Magnitude
    window_features+=[np.max(x) for x in window_fft]

    # Spectral Centroid
    window_features+=[freq_cent(x, np.fft.fftfreq(window.shape[0])) for x in window_fft]
    
    # Total Signal Power
    window_features+=[np.sum(x**2) for x in window_fft]
    
    train_features[i] = window_features

    i += 1
    print(f"train window: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    
    # if i == 1:
    #   print(f"window: {i}/{len(train_windows)}, feature len: {len(window_features)}") 
    #   print(f"window features: ")
    #   print(window_features)
    #   print(f"train features: ")
    #   print(train_features)
    # if i == 100:
    #   print(f"\nwindow: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    #   print(train_features)
    # if i == 500:
    #   print(f"\nwindow: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    # if i == 1000:
    #   print(f"\nwindow: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    # if i == 10000:
    #   print(f"\nwindow: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    # if i == 50000:
    #   print(f"\nwindow: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    # if i == 100000:
    #   print(f"window: {i}/{len(train_windows)}, feature len: {len(window_features)}")
    #   #break
    # if i == 1000000:
    #   print(f"window: {i}/{len(train_windows)}, feature len: {len(window_features)}")

# GENERATE TEST FEATURES
i=0
for window in test_windows:
    window_features = []

    b_mag = (np.linalg.norm(window[:3,:],axis=0))
    t_mag = (np.linalg.norm(window[3:,:],axis=0))

    b_mag = b_mag.reshape(list(b_mag.shape)+[1])
    t_mag = t_mag.reshape(list(t_mag.shape)+[1])

    window = np.transpose(window)

    window_withMag = np.concatenate([window,b_mag,t_mag], axis=1)
    
    window_withMag = np.transpose(window_withMag)

    #  Filtering
    # create the LP filter
    # 1Hz 4th order LP Butterworth filter
    b, a  = signal.butter(4, Wn=1, fs=fs, btype='lowpass')

    # create the HP filter
    # 1Hz 4th order HP Butterworth filter
    d, c = signal.butter(4, Wn=1, fs=fs, btype='highpass')

    window_grav = signal.filtfilt(b, a, window_withMag, axis=1, padlen=10)
    window_mot = signal.filtfilt(d, c, window_withMag, axis=1, padlen=10)

    #print(len(window_features))
    # kept these as .append instead of += bc when I tested it, it threw an error when including the means later on
    window_features.append(corr_grav(window_withMag[0],window_withMag[1]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[2]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[3]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[0],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[1],window_withMag[2]))
    window_features.append(corr_grav(window_withMag[1],window_withMag[3]))
    window_features.append(corr_grav(window_withMag[1],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[1],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[2],window_withMag[3]))
    window_features.append(corr_grav(window_withMag[2],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[2],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[3],window_withMag[4]))
    window_features.append(corr_grav(window_withMag[3],window_withMag[5]))
    window_features.append(corr_grav(window_withMag[4],window_withMag[5]))

    window_features.append(corr_grav(window_withMag[6],window_withMag[7]))

    # GRAVITY:
    #  Mean over all axes
    window_features.append(np.mean(window_grav))

    # Mean
    window_features+=[np.mean(x) for x in window_grav]

    # Median
    window_features+=[np.median(x) for x in window_grav]

    # Standard Deviation
    window_features+=[np.std(x) for x in window_grav]

    #  Coefficient of Variation
    window_features+=[cv_gravity(x) for x in window_grav]
    
    #  Min Value
    window_features+=[np.min(x) for x in window_grav]

    # 25th Percentile
    window_features+=[np.percentile(x,25) for x in window_grav]

    # 50th Percentile
    window_features+=[np.percentile(x,50) for x in window_grav]

    # 75th Percentile
    window_features+=[np.percentile(x,75) for x in window_grav]

    # Max Value
    window_features+=[np.max(x) for x in window_grav]

    # ACCELERATION:
    # Skew
    window_features+=[stats.skew(x) for x in window_mot]

    # Kurtosis
    window_features+=[stats.kurtosis(x) for x in window_mot]

    # Signal Energy
    window_features+=[np.sum(x**2) for x in window_mot]

    # Frequency-Domain Mean
    # FFT
    window_fft = np.abs(np.fft.fft(window_withMag, axis=0))

    window_features+=[np.mean(x) for x in window_fft]

    # Frequency-Domain Standard Deviation
    window_features+=[np.std(x) for x in window_fft]

    # Dominant Frequency
    window_features+=[np.fft.fftfreq(window.shape[0])[x.argmax(axis=0)] for x in window_fft]

    # Dominant Frequency Magnitude
    window_features+=[np.max(x) for x in window_fft]

    # Spectral Centroid
    window_features+=[freq_cent(x, np.fft.fftfreq(window.shape[0])) for x in window_fft]
    
    # Total Signal Power
    window_features+=[np.sum(x**2) for x in window_fft]
    
    test_features[i] = window_features
    
    i+=1
    print(f"test window: {i}/{len(test_windows)}, feature len: {len(window_features)}")


    
# Replace NaN values with 0
train_features[np.isnan(train_features)] = 0
test_features[np.isnan(test_features)] = 0

# SAVE FEATURES
train_features.flush()
test_features.flush()
