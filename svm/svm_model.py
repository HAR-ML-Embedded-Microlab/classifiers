from libsvm.svmutil import *
import numpy as np

train_features = np.load('work/train_features.npy', mmap_mode='r')
train_labels = np.load('work/train_labels.npy', mmap_mode='r')
test_features = np.load('work/test_features.npy', mmap_mode='r')
test_labels = np.load('work/test_labels.npy', mmap_mode='r')

m = svm_train(train_labels, train_features, '-s 0 -t 2 -c 10')
svm_save_model("work/model_svm_trained.model")

(p_labels, p_acc, p_vals) = svm_predict(test_labels, test_features)
print(p_acc)

