import os
import numpy as np
# from utils import laserscan
import yaml
from torch.utils.data import Dataset
import torch
# import spconv
import math
from tqdm import tqdm

config_file = os.path.join('Data/semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]

LABELS_REMAP = kitti_config["learning_map"]
LABEL_INV_REMAP = kitti_config["learning_map_inv"]




SPLIT_SEQUENCES = {
    "train": ["00", "02", "03", "05", "06", "07", "09", "10"],
    "valid": ["08"],
    "test": ["11"]
    # "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
    # "valid": ["08"],
    # "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
}


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed



class KittiDataset(Dataset):
    """Kitti Dataset for 3D mapping project
    
    Access to the processed data, including evaluation labels predictions velodyne poses times
    """

    def __init__(self, directory,
            device='cuda',
            num_frames=4,
            past_frames=10,
            split='train',
            transform_pose=False,
            get_gt = True,
            grid_size = [256,256,13],
            grid_size_train = [256,256,32]
            ):
        
        self.get_gt = get_gt
        self._grid_size = grid_size
        self.grid_dims = np.asarray(self._grid_size)
        self._grid_dims_train = np.asarray(grid_size_train)
        self._eval_size = list(np.uint32(self._grid_size))
        self.coor_ranges = [-25.6,-25.6,-6.4] + [25.6,25.6,6.4]
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_dims_train[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_dims_train[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_dims_train[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

        self._directory = os.path.join(directory, 'sequences')
        self._num_frames = num_frames
        self._past_frames = past_frames
        self.device = device
        self.split = split
        self.transform_pose = transform_pose

        self._velodyne_list = []
        self._frames_list = []
        self._bev_labels = []
        self._poses = np.empty((0,12))
        self._Tr = np.empty((0,12))

        self._num_frames_scene = []

        self._seqs = SPLIT_SEQUENCES[self.split]


        for seq in self._seqs:
            velodyne_dir = os.path.join(self._directory, seq, 'velodyne')
            bev_dir = os.path.join(self._directory, seq, 'bev_gt2')
            self._num_frames_scene.append(len(os.listdir(velodyne_dir)))
            frames_list = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(velodyne_dir))]
            self._frames_list.extend(frames_list)
            self._velodyne_list.extend([os.path.join(velodyne_dir, str(frame).zfill(6)+'.bin') for frame in frames_list])
            self._bev_labels.extend([os.path.join(bev_dir, str(frame).zfill(6) + '.bin') for frame in frames_list])

            pose = np.loadtxt(os.path.join(self._directory, seq, 'poses.txt'))
            Tr = np.genfromtxt(os.path.join(self._directory, seq, 'calib.txt'))[-1,1:]
            Tr = np.repeat(np.expand_dims(Tr, axis=1).T,pose.shape[0],axis=0)
            self._Tr = np.vstack((self._Tr, Tr))
            self._poses = np.vstack((self._poses, pose))


        assert len(self._velodyne_list) == np.sum(self._num_frames_scene), f"inconsitent number of frames detected, check the dataset"
        self._poses = self._poses.reshape(sum(self._num_frames_scene), 12)
        self._Tr = self._Tr.reshape(sum(self._num_frames_scene), 12)



    def collate_fn(self, data):
        input_batch = [bi[0] for bi in data]
        if not self.get_gt:
            fnames_batch = [bi[1] for bi in data]
            return input_batch, fnames_batch
        output_batch = [bi[1] for bi in data]
        return input_batch, output_batch
    
    # Use all frames, if there is no data then zero pad
    def __len__(self):
        return sum(self._num_frames_scene)
    


    def get_pose(self, idx):
        pose = np.zeros((4, 4))
        pose[3, 3] = 1
        pose[:3, :4] = self._poses[idx,:].reshape(3, 4)

        Tr = np.zeros((4, 4))
        Tr[3, 3] = 1
        Tr[:3, :4] = self._Tr[idx,:].reshape(3,4)

        Tr = Tr.astype(np.float32)
        pose = pose.astype(np.float32)
        global_pose = np.matmul(np.linalg.inv(Tr), np.matmul(pose, Tr))

        return global_pose

    def __getitem__(self, idx):
        # -1 indicates no data
        # the final index is the output
        idx_range = self.find_horizon(idx)
        if self.transform_pose:
            ego_pose = self.get_pose(idx_range[-1])
            to_ego = np.linalg.inv(ego_pose)
         
        current_horizon = np.zeros((idx_range.shape[0], int(self._grid_dims_train[0]), int(self._grid_dims_train[1]), int(self._grid_dims_train[2])), dtype=np.float32)

        t_i = 0
        for i in idx_range:
            if i == -1: # Zero pad
                points = np.zeros((1, 3), dtype=np.float32)
                
            else:
                idx_2 = self.find_horizon_2(i)
                
                if len(idx_2) == 1:

                    points = np.fromfile(self._velodyne_list[i],dtype=np.float32).reshape(-1,4)[:, :3]
                    if self.transform_pose:
                        to_world = self.get_pose(i)
                        relative_pose = np.matmul(to_ego, to_world)
                        points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]

                
                else:
                    pose_current = self.get_pose(i)
                    to_current = np.linalg.inv(pose_current)
                    points = np.array([]).reshape(0, 3)
                    for j in idx_2:
                        points_t = np.fromfile(self._velodyne_list[j],dtype=np.float32).reshape(-1,4)[:, :3]
                        if self.transform_pose:
                            to_world = self.get_pose(j)
                            relative_pose = np.matmul(to_current, to_world)
                            points_t = np.dot(relative_pose[:3, :3], points_t.T).T + relative_pose[:3, 3]
                        points = np.vstack((points, points_t))
                    
                    if self.transform_pose:
                        to_world = self.get_pose(i)
                        relative_pose = np.matmul(to_ego, to_world)
                        points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]
                    
            current_horizon = self.points_to_voxels(current_horizon, points, t_i)

            t_i += 1
            
        if self.get_gt:
            output = np.fromfile(self._bev_labels[idx_range[-1]], dtype=np.uint8).reshape(self._eval_size[0], self._eval_size[1])
        else:
            output = None
            fname = self._velodyne_list[i]
            return current_horizon, fname
            


        return current_horizon, output
        
            
    def find_horizon(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._num_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._num_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range

    def find_horizon_2(self, idx):
        end_idx = idx
        idx_range = np.arange(idx-self._past_frames, idx)+1
        diffs = np.asarray([int(self._frames_list[end_idx]) - int(self._frames_list[i]) for i in idx_range])
        good_difs = -1 * (np.arange(-self._past_frames, 0) + 1)
        
        idx_range[good_difs != diffs] = -1

        return idx_range
    
    def points_to_voxels(self, voxel_grid, points, t_i):
        # Valid voxels (make sure to clip)
        valid_point_mask= np.all(
            (points < self.max_bound) & (points >= self.min_bound), axis=1)
        valid_points = points[valid_point_mask, :]
        voxels = np.floor((valid_points - self.min_bound) / self.voxel_sizes).astype(int)
        # Clamp to account for any floating point errors
        maxes = np.reshape(self._grid_dims_train - 1, (1, 3))
        mins = np.zeros_like(maxes)
        voxels = np.clip(voxels, mins, maxes).astype(int)
        
        voxel_grid[t_i, voxels[:, 0], voxels[:, 1], voxels[:, 2]] += 1

        return voxel_grid

