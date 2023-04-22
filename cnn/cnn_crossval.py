import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import json
import seaborn as sns

model = models.load_model('work/model_trained')

test_windows = np.load('work/test_windows.npy', mmap_mode='r')
test_labels = np.load('work/test_labels.npy', mmap_mode='r')

with tf.device('CPU'):
    test_dataset = tf.data.Dataset.from_tensor_slices((test_windows, test_labels)).batch(64)

with open('work/class_weights.json', 'r') as jsonfile:
    class_weights = json.load(jsonfile)
    # Cast to correct format
    class_weights = {int(k):float(v) for k,v in class_weights.items()}

label_names = [
    "walking",
    "running",
    "shuffling", 
    "stairs-up", 
    "stairs-down", 
    "standing", 
    "sitting", 
    "lying", 
    "cycling (sit)", 
    "cycling (stand)", 
    "cycling (sit-in)", 
    "cycling (stand-in)"
]

test_preds = tf.argmax(model.predict(test_dataset), axis=1)

met_accuracy = tf.keras.metrics.Accuracy()
met_precision = tf.keras.metrics.Precision()
met_recall = tf.keras.metrics.Recall()

met_accuracy.update_state(test_labels,test_preds)
met_precision.update_state(test_labels,test_preds)
met_recall.update_state(test_labels,test_preds)

print("Accuracy: ",met_accuracy.result().numpy())
print("Precision: ",met_precision.result().numpy())
print("Recall: ",met_recall.result().numpy())

weights = tf.map_fn(fn=lambda t:class_weights[t], elems=test_preds)
cmx = tf.math.confusion_matrix(test_labels, test_preds, weights)

plt.figure(figsize=(10, 8))
sns.heatmap(cmx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()


