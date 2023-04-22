import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import json

model = models.load_model('work/model_trained')

with open('work/class_weights.json', 'r') as jsonfile:
    class_weights = json.load(jsonfile)
    # Cast to correct format
    class_weights = {int(k):float(v) for k,v in class_weights.items()}


train_windows = np.load('work/train_windows.npy', mmap_mode='r')
train_labels = np.load('work/train_labels.npy', mmap_mode='r')
test_windows = np.load('work/test_windows.npy', mmap_mode='r')
test_labels = np.load('work/test_labels.npy', mmap_mode='r')

with tf.device('CPU'):
    train_dataset = tf.data.Dataset.from_tensor_slices((train_windows, train_labels)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_windows, test_labels)).batch(64)

# Todo: figure out how to save training history data
hist = model.fit(train_dataset, epochs=10, validation_data=test_dataset, class_weight=class_weights)
model.save("work/model_trained")
with open("work/model_trained_hist","w") as file:
    pickle.dump(hist.history, file)
