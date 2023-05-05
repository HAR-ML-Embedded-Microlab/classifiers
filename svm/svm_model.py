from libsvm.svmutil import *
import numpy as np
from numpy.lib.format import open_memmap

train_features = np.load('/home/ramona_rajagopalan/classifiers/work/train_features.npy', mmap_mode='r')
train_labels = np.load('/home/ramona_rajagopalan/classifiers/work/train_labels.npy', mmap_mode='r')
test_features = np.load('/home/ramona_rajagopalan/classifiers/work/test_features.npy', mmap_mode='r')
test_labels = np.load('/home/ramona_rajagopalan/classifiers/work/test_labels.npy', mmap_mode='r')

train_labels_reone = open_memmap('/home/ramona_rajagopalan/classifiers/work/train_labels_reone.npy', mode='w+', dtype=np.int32, shape=(train_labels.shape[0],))
test_labels_reone = open_memmap('/home/ramona_rajagopalan/classifiers/work/test_labels_reone.npy', mode='w+', dtype=np.int32, shape=(test_labels.shape[0],))

# print(train_labels[:5])

# Reverse one hot encoding
train_labels_reone[:] = [i[0] for i in train_labels]
test_labels_reone[:] = [i[0] for i in test_labels]

# print(train_labels_reone[:5])
# print(train_labels_reone.shape)

# Train
m = svm_train(train_labels_reone, train_features, '-s 0 -t 2 -c 10')
svm_save_model("/home/ramona_rajagopalan/classifiers/work/model_svm_trained.model",m)

(p_labels, p_acc, p_vals) = svm_predict(test_labels_reone, test_features, m)
print(p_acc)