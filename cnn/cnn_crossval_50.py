import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import json
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

model = models.load_model('point-50-sec/cnn_model')

test_windows = np.load('point-50-sec/test_windows.npy', mmap_mode='r')
test_labels_reone = np.load('point-50-sec/test_labels_reone.npy', mmap_mode='r')

with tf.device('CPU'):
    test_dataset = tf.data.Dataset.from_tensor_slices((test_windows, test_labels_reone)).batch(8192)

label_names = [
    "walking",
    "running",
    "shuffling", 
    "stairs-up", 
    "stairs-down", 
    "standing", 
    "sitting", 
    "lying", 
    "cycling\n(sit)", 
    "cycling\n(stand)", 
    "cycling\n(sit-in)", 
    "cycling\n(stand-in)"
]

test_preds = tf.argmax(model.predict(test_dataset), axis=1)

# met_accuracy = tf.keras.metrics.Accuracy()
# met_precision = tf.keras.metrics.Precision()
# met_recall = tf.keras.metrics.Recall()

# met_accuracy.update_state(test_labels_reone,test_preds)
# met_precision.update_state(test_labels_reone,test_preds)
# met_recall.update_state(test_labels_reone,test_preds)

# print("Accuracy: ",met_accuracy.result().numpy())
# print("Precision: ",met_precision.result().numpy())
# print("Recall: ",met_recall.result().numpy())

# cmx = tf.math.confusion_matrix(test_labels_reone, test_preds)

# plt.figure(figsize=(15, 15))
# sns.heatmap(cmx,
#             xticklabels=label_names,
#             yticklabels=label_names,
#             annot=True, fmt='g')
# plt.xlabel('Prediction')
# plt.ylabel('Label')
# # plt.show()
# plt.savefig('point-50-sec/cmx_50.png', dpi=300)

title = "Normalized CNN Confusion Matrix (0.50s)"

disp = ConfusionMatrixDisplay.from_predictions(
    test_labels_reone,
    test_preds,
    display_labels=label_names,
    cmap=plt.cm.Blues,
    normalize='true',
)
disp.ax_.set_title(title)

disp.figure_.set_figwidth(20)
disp.figure_.set_figheight(15) 

disp.figure_.savefig('point-50-sec/cmx_50_norm.png', dpi=300)


