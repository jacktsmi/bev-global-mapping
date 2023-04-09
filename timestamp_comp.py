import os
import numpy as np


def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


velodyne_timestamp_path = '/home/jason/Downloads/2011_09_30_drive_0028_sync/2011_09_30/2011_09_30_drive_0028_sync/velodyne_points/timestamps.txt'
imu_timestamp_path = '/home/jason/Downloads/2011_09_30_drive_0028_extract/2011_09_30/2011_09_30_drive_0028_extract/oxts/timestamps.txt'


# velodyne timestamps
velodyne_timestamp_list_path = open(velodyne_timestamp_path, 'r')
velodyne_timestamp_list_og = velodyne_timestamp_list_path.readlines()

velodyne_timestamp = np.zeros((len(velodyne_timestamp_list_og)))
c = 0
for line in velodyne_timestamp_list_og:
    timestamp_t = line.split(' ')[-1]
    hr, minute, sec = timestamp_t.split(":")
    timestamp_s = int(hr)*3600.0 + int(minute)*60.0 + float(sec)
    velodyne_timestamp[c] = timestamp_s
    c += 1


# imu timestamps
imu_timestamp_list_path = open(imu_timestamp_path, 'r')
imu_timestamp_list_og = imu_timestamp_list_path.readlines()

imu_timestamp = np.zeros((len(imu_timestamp_list_og)))
c = 0
for line in imu_timestamp_list_og:
    timestamp_t = line.split(' ')[-1]
    hr, minute, sec = timestamp_t.split(":")
    timestamp_s = int(hr)*3600.0 + int(minute)*60.0 + float(sec)
    imu_timestamp[c] = timestamp_s
    c += 1

closest_idx = closest_argmin(velodyne_timestamp, imu_timestamp)

save_dir = './closest_idx.txt'

np.savetxt(save_dir, closest_idx, fmt='%d')

