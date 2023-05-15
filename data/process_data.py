import random
import json

import pandas as pd
import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import sliding_window_view
from numpy.lib.format import open_memmap
from keras.utils.np_utils import to_categorical

num_samples = 50 # 1s

harth_filenames = [
    './harth-ml-experiments/harth/S006.csv',
    './harth-ml-experiments/harth/S008.csv',
    './harth-ml-experiments/harth/S009.csv',
    './harth-ml-experiments/harth/S010.csv',
    './harth-ml-experiments/harth/S012.csv',
    './harth-ml-experiments/harth/S013.csv',
    './harth-ml-experiments/harth/S014.csv',
    './harth-ml-experiments/harth/S015.csv',
    './harth-ml-experiments/harth/S016.csv',
    './harth-ml-experiments/harth/S017.csv',
    './harth-ml-experiments/harth/S018.csv',
    './harth-ml-experiments/harth/S019.csv',
    './harth-ml-experiments/harth/S020.csv',
    './harth-ml-experiments/harth/S021.csv',
    './harth-ml-experiments/harth/S022.csv',
    './harth-ml-experiments/harth/S023.csv',
    './harth-ml-experiments/harth/S024.csv',
    './harth-ml-experiments/harth/S025.csv',
    './harth-ml-experiments/harth/S026.csv',
    './harth-ml-experiments/harth/S027.csv',
    './harth-ml-experiments/harth/S028.csv',
    './harth-ml-experiments/harth/S029.csv',
    './harth-ml-experiments/har70plus/501.csv',
    './harth-ml-experiments/har70plus/502.csv',
    './harth-ml-experiments/har70plus/503.csv',
    './harth-ml-experiments/har70plus/504.csv',
    './harth-ml-experiments/har70plus/505.csv',
    './harth-ml-experiments/har70plus/506.csv',
    './harth-ml-experiments/har70plus/507.csv',
    './harth-ml-experiments/har70plus/508.csv',
    './harth-ml-experiments/har70plus/509.csv',
    './harth-ml-experiments/har70plus/510.csv',
    './harth-ml-experiments/har70plus/511.csv',
    './harth-ml-experiments/har70plus/512.csv',
    './harth-ml-experiments/har70plus/513.csv',
    './harth-ml-experiments/har70plus/514.csv',
    './harth-ml-experiments/har70plus/515.csv',
    './harth-ml-experiments/har70plus/516.csv',
    './harth-ml-experiments/har70plus/517.csv',
    './harth-ml-experiments/har70plus/518.csv'
]

permanent_shuffling = [19, 12, 14, 15, 1, 10, 4, 18, 5, 20, 21, 6, 11, 16, 13, 9, 7, 17, 38, 29, 33, 27, 37, 26, 40, 35, 31, 32, 25, 23, 28, 34, 2, 3, 8, 22, 24, 30, 36, 39]

harth_filenames = [harth_filenames[i-1] for i in permanent_shuffling]
train_test_split = int(0.8 * len(harth_filenames))

num_windows_train = 0
num_windows_test = 0

print("Ingesting CSV files...")

data_nps = [None] * len(permanent_shuffling)
for i in range(len(permanent_shuffling)):
    data_df = pd.read_csv(harth_filenames[i])
    data_df.drop(['timestamp'], axis=1, inplace=True)
    data_df.drop(['index'], axis=1, errors='ignore', inplace=True)
    data_df.drop([''], axis=1, errors='ignore', inplace=True)
    data_df.drop([' '], axis=1, errors='ignore', inplace=True)
    data_df.rename({"Unnamed: 0":"a"}, errors='ignore', axis="columns", inplace=True)
    data_df.drop(['a'], axis=1, errors='ignore', inplace=True)
    data_df.dropna(inplace=True)
    data_nps[i] = data_df.to_numpy(dtype=np.float32)
    if permanent_shuffling[i]==1:
        data_nps[i]=data_nps[i][::2]
    if i < train_test_split:
        num_windows_train += data_nps[i].shape[0] - num_samples + 1
    else:
        num_windows_test += data_nps[i].shape[0] - num_samples + 1
    del data_df # memory is scarce after all
    
print("Processing windows...")

train_windows = open_memmap('1-sec/train_windows_tmp.npy', mode='w+', dtype=np.float32, shape=(num_windows_train, 6, num_samples))
train_labels = open_memmap('1-sec/train_labels_tmp.npy', mode='w+', dtype=np.int32, shape=(num_windows_train,))
test_windows = open_memmap('1-sec/test_windows_tmp.npy', mode='w+', dtype=np.float32, shape=(num_windows_test, 6, num_samples))
test_labels = open_memmap('1-sec/test_labels_tmp.npy', mode='w+', dtype=np.int32, shape=(num_windows_test,))

num_windows_train_processed = 0
num_windows_test_processed = 0

for i in range(len(permanent_shuffling)):
    # Since the sensors have been set to +-8G, divide by 8 to scale to 0~1 range.
    if i < train_test_split:
        train_windows[num_windows_train_processed:num_windows_train_processed+data_nps[i].shape[0] - num_samples + 1] = sliding_window_view(data_nps[i][:,0:6], window_shape=num_samples, axis=0)/np.float32(8)
        train_labels[num_windows_train_processed:num_windows_train_processed+data_nps[i].shape[0] - num_samples + 1] = sp.stats.mode(sliding_window_view(data_nps[i][:,6], window_shape=num_samples), keepdims=False, axis=1)[0].astype(np.int32)
        num_windows_train_processed += data_nps[i].shape[0] - num_samples + 1
    else:
        test_windows[num_windows_test_processed:num_windows_test_processed+data_nps[i].shape[0] - num_samples + 1] = sliding_window_view(data_nps[i][:,0:6], window_shape=num_samples, axis=0)/np.float32(8)
        test_labels[num_windows_test_processed:num_windows_test_processed+data_nps[i].shape[0] - num_samples + 1] = sp.stats.mode(sliding_window_view(data_nps[i][:,6], window_shape=num_samples), keepdims=False, axis=1)[0].astype(np.int32)
        num_windows_test_processed += data_nps[i].shape[0] - num_samples + 1

del data_nps[:]
del data_nps

print("Relabelling data...")

train_labels[train_labels==13] = 9
train_labels[train_labels==14] = 10
train_labels[train_labels==130] = 11
train_labels[train_labels==140] = 12
train_labels[:] = train_labels-1

test_labels[test_labels==13] = 9
test_labels[test_labels==14] = 10
test_labels[test_labels==130] = 11
test_labels[test_labels==140] = 12
test_labels[:] = test_labels-1

print("Saving data...")

train_labels.flush()
train_windows.flush()
test_labels.flush()
test_windows.flush()

print("Generating class weights")

# Use sklearn.utils.class_weight.compute_class_weight's algorithm without importing sklearn.
num_classes = 12
weights = train_labels.size/(num_classes*np.bincount(train_labels))
class_weights = {i:weights[i] for i in range(num_classes)}

with open("1-sec/class_weights.json", "w") as jsonfile:
    json.dump(class_weights, jsonfile, indent=4)   
    
print("Shuffling dataset...")

train_windows_shuffled = open_memmap('1-sec/train_windows.npy', mode='w+', dtype=np.float32, shape=(num_windows_train, 6, num_samples))
train_labels_shuffled = open_memmap('1-sec/train_labels.npy', mode='w+', dtype=np.int32, shape=(num_windows_train,))
test_windows_shuffled = open_memmap('1-sec/test_windows.npy', mode='w+', dtype=np.float32, shape=(num_windows_test, 6, num_samples))
test_labels_shuffled = open_memmap('1-sec/test_labels.npy', mode='w+', dtype=np.int32, shape=(num_windows_test,))

train_shuffle = np.arange(train_labels.size)
test_shuffle = np.arange(test_labels.size)
np.random.shuffle(train_shuffle)
np.random.shuffle(test_shuffle)

for i in range(train_labels.size):
    train_windows_shuffled[i] = train_windows[train_shuffle[i]]
    train_labels_shuffled[i] = train_labels[train_shuffle[i]]
    
for i in range(test_labels.size):
    test_windows_shuffled[i] = test_windows[test_shuffle[i]]
    test_labels_shuffled[i] = test_labels[test_shuffle[i]]

print("Flush...")

train_windows_shuffled.flush()
train_labels_shuffled.flush()
test_windows_shuffled.flush()
test_labels_shuffled.flush()

print("Preprocessing Complete!")
