# adding .npy files to evaluation/ folder before running evaluation
import numpy as np
import matplotlib.pyplot as plt

data_gt = np.load('evaluation/KITTI_Scene08_GT_Labels_new.npy')
data_test = np.load('evaluation/KITTI_Scene08_Test_Labels_new.npy')
size_gt = data_gt.shape

correct_num = np.zeros(20)
total_num = np.zeros(20)
fp = np.zeros(20)
fn = np.zeros(20)
IoU = np.zeros(20)
precision = np.zeros(20)
recall = np.zeros(20)
total_num[0] = data_gt.size  # 0 = overall IoU


for i in range(size_gt[0]):
    for j in range(size_gt[1]):
        label_gt = data_gt[i][j]
        label_test = data_test[i][j]

        if label_gt == label_test:
            total_num[label_gt] += 1
            correct_num[label_gt] += 1
            correct_num[0] += 1
        else:
            total_num[label_gt] += 1
            total_num[label_test] += 1
            fp[label_test] += 1
            fn[label_gt] += 1


for i in range(len(correct_num)):
    if (total_num[i] != 0 and correct_num[i] != 0):
        IoU[i] = correct_num[i]/total_num[i]
        precision[i] = correct_num[i]/(correct_num[i] + fp[i])
        recall[i] = correct_num[i]/(correct_num[i]+fn[i])
        
print('IoU','Precision','Recall')
for i in range(len(correct_num)):
    if (total_num[i] != 0 and correct_num[i] != 0):
        
        print(i, IoU[i], precision[i], recall[i])

