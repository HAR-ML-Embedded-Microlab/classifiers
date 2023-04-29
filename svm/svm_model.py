from libsvm.svmutil import *

train_windows = np.load('work/train_windows.npy', mmap_mode='r')
train_labels = np.load('work/train_labels.npy', mmap_mode='r')
test_windows = np.load('work/test_windows.npy', mmap_mode='r')
test_labels = np.load('work/test_labels.npy', mmap_mode='r')

m = svm_train(train_labels, train_windows, '-s 0 -t 2 -c 10')
svm_save_model("work/model_svm_trained.model")

(p_labels, p_acc, p_vals) = svm_predict(test_labels, test_windows)
print(p_acc)

