import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import json
from numpy.lib.format import open_memmap

model = models.load_model('/home/ramona_rajagopalan/classifiers/point-75-sec/cnn_model')

with open('/home/ramona_rajagopalan/classifiers/point-75-sec/class_weights.json', 'r') as jsonfile:
    class_weights = json.load(jsonfile)
    # Cast to correct format
    class_weights = {int(k):float(v) for k,v in class_weights.items()}

train_windows = np.load('/home/ramona_rajagopalan/classifiers/point-75-sec/train_windows.npy', mmap_mode='r')
train_labels = np.load('/home/ramona_rajagopalan/classifiers/point-75-sec/train_labels.npy', mmap_mode='r')
test_windows = np.load('/home/ramona_rajagopalan/classifiers/point-75-sec/test_windows.npy', mmap_mode='r')
test_labels = np.load('/home/ramona_rajagopalan/classifiers/point-75-sec/test_labels.npy', mmap_mode='r')

train_labels_reone = open_memmap('/home/ramona_rajagopalan/classifiers/point-75-sec/train_labels_reone.npy', mode='w+', dtype=np.int32, shape=(train_labels.shape[0],))
test_labels_reone = open_memmap('/home/ramona_rajagopalan/classifiers/point-75-sec/test_labels_reone.npy', mode='w+', dtype=np.int32, shape=(test_labels.shape[0],))

# Reverse one hot encoding
train_labels_reone[:] = [i[0] for i in train_labels]
test_labels_reone[:] = [i[0] for i in test_labels]

with tf.device('CPU'):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_windows, train_labels_reone)).batch(8192)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_windows, test_labels_reone)).batch(8192)

# Todo: figure out how to save training history data
hist = model.fit(train_dataset, epochs=80, validation_data=test_dataset, verbose=2, class_weight=class_weights)
model.save("/home/ramona_rajagopalan/classifiers/point-75-sec/cnn_model")
with open("/home/ramona_rajagopalan/classifiers/point-75-sec/cnn_model_hist","wb") as file:
    pickle.dump(hist.history, file)
