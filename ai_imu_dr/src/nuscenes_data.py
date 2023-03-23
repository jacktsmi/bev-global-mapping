from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from termcolor import cprint
import pickle
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import json
from utils_numpy_filter import NUMPYIEKF as IEKF
from scipy.spatial.transform import Rotation

class NuScenesParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-2
    cov_acc = 1e-2
    cov_b_omega = 1e-2
    cov_b_acc = 1e-2
    cov_Rot_c_i = 1e-2
    cov_t_c_i = 1e-2
    cov_Rot0 = 1e-1
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-2
    cov_b_acc0 = 1e-2
    cov_Rot_c_i0 = 1e-2
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(NuScenesParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(NuScenesParameters) if
                     not a.startswith('__') and not callable(getattr(NuScenesParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(NuScenesParameters, attr))

class NuScenesData(Dataset):
    pickle_extension = ".p"
    """extension of the file saved in pickle format"""
    file_normalize_factor = "normalize_factors.p"
    """name of file for normalizing input"""

    def __init__(self, args):
        # paths
        self.path_data_save = args.path_data_save
        """path where data are saved"""
        self.path_results = args.path_results
        """path to the results"""
        self.path_temp = args.path_temp
        """path for temporary files"""

        self.datasets_test = args.test_sequences
        """test datasets"""
        self.datasets_validation = args.cross_validation_sequences
        """cross-validation datasets"""
        
        self.n_train = args.n_train # Number training files
        self.test_scene = args.test_scene
        
        self.data = [] # List of dicts

        # names of the sequences
        self.datasets = []
        """dataset names"""
        self.datasets_train = []
        """train datasets"""

        self.datasets_validatation_filter = OrderedDict()
        """Validation dataset with index for starting/ending"""
        self.datasets_train_filter = OrderedDict()
        """Validation dataset with index for starting/ending"""
        
        self.sigma_gyro = 1.e-4
        self.sigma_acc = 1.e-4
        self.sigma_b_gyro = 1.e-5
        self.sigma_b_acc = 1.e-4

        # factors for normalizing inputs
        self.normalize_factors = None
        self.get_datasets()
        self.set_normalize_factors()

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.datasets)

    def helper(self, scene_num, is_train):
        imu_measurement_file = self.path_data_save + f'/scene-{scene_num:0>4}_ms_imu.json'
        gt_file = self.path_data_save + f'/scene-{scene_num:0>4}_pose.json'
        self.datasets.append(f'scene-{scene_num:0>4}_ms_imu.json')
        
        if is_train:
            self.datasets_train_filter[f'scene-{scene_num:0>4}_ms_imu.json'] = [0, 1800]
        
        with open(imu_measurement_file) as imu, open(gt_file) as gt:
            imu_list = json.load(imu)
            gt_list = json.load(gt)
        
        # Convert list of dicts into dict of lists
        # Times aren't aligned for ground truth and measurements, so find nearest time
        data_dict = {'t': [],
                     'ang_gt': [],
                     'p_gt': [],
                     'v_gt': [],
                     'u': []}
        
        gt_times = [gt_list[i]['utime'] for i in range(len(gt_list))]

        start_time = 0
        for i in range(len(imu_list)):
            t = imu_list[i]['utime']
            gt_nearest_idx = find_nearest(gt_times, t)
            gt_t = gt_list[gt_nearest_idx]['utime']
            q = gt_list[gt_nearest_idx]['orientation']
            r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            roll, pitch, yaw = IEKF.to_rpy(r.as_matrix())
            ang_gt = [roll, pitch, yaw]
            p_gt = gt_list[gt_nearest_idx]['pos']
            v_gt = gt_list[gt_nearest_idx]['vel']
            u = imu_list[i]['rotation_rate'] + imu_list[i]['linear_accel']
            # u[3:] += np.array([0, 0, -9.80655])
            t = (gt_t + t) / 2
            if i == 0:
                start_time = t
            
            data_dict['t'].append((gt_t - start_time) / 1e6)
            data_dict['ang_gt'].append(ang_gt)
            data_dict['p_gt'].append(p_gt)
            data_dict['v_gt'].append(v_gt)
            data_dict['u'].append(u)
        
        data_dict['t'] = torch.from_numpy(np.array(data_dict['t'])).float()
        data_dict['ang_gt'] = torch.from_numpy(np.array(data_dict['ang_gt'])).float()
        data_dict['p_gt'] = torch.from_numpy(np.array(data_dict['p_gt'])).float()
        data_dict['v_gt'] = torch.from_numpy(np.array(data_dict['v_gt'])).float()
        data_dict['u'] = torch.from_numpy(np.array(data_dict['u'])).float()
        
        self.data.append(data_dict)
        self.dump(data_dict,imu_measurement_file)

    def get_datasets(self):
        self.helper(self.test_scene, is_train=False)
        for i_tr in range(self.n_train):
            self.helper(i_tr+1, is_train=True)
        self.divide_datasets()

    def divide_datasets(self):
        for dataset in self.datasets:
            if (not dataset in self.datasets_test) and (not dataset in self.datasets_validation):
                self.datasets_train += [dataset]

    def dataset_name(self, i):
        return self.datasets[i]

    def get_data(self, i):
        pickle_dict = self[self.datasets.index(i) if type(i) != int else i]
        return pickle_dict['t'], pickle_dict['ang_gt'], pickle_dict['p_gt'], pickle_dict['v_gt'],\
               pickle_dict['u']

    def set_normalize_factors(self):
        path_normalize_factor = os.path.join(self.path_temp, self.file_normalize_factor)
        # we set factors only if file does not exist
        if os.path.isfile(path_normalize_factor):
            pickle_dict = self.load(path_normalize_factor)
            self.normalize_factors = pickle_dict['normalize_factors']
            self.num_data = pickle_dict['num_data']
            return

        # first compute mean
        self.num_data = 0

        for i, dataset in enumerate(self.datasets_train):
            pickle_dict = self.load(self.path_data_save, dataset)
            u = pickle_dict['u']
            if i == 0:
                u_loc = u.sum(dim=0)
            else:
                u_loc += u.sum(dim=0)
            self.num_data += u.shape[0]
        u_loc = u_loc / self.num_data

        # second compute standard deviation
        for i, dataset in enumerate(self.datasets_train):
            pickle_dict = self.load(self.path_data_save, dataset)
            u = pickle_dict['u']
            if i == 0:
                u_std = ((u - u_loc) ** 2).sum(dim=0)
            else:
                u_std += ((u - u_loc) ** 2).sum(dim=0)
        u_std = (u_std / self.num_data).sqrt()

        self.normalize_factors = {
            'u_loc': u_loc, 'u_std': u_std,
            }
        print('... ended computing normalizing factors')
        pickle_dict = {
            'normalize_factors': self.normalize_factors, 'num_data': self.num_data}
        self.dump(pickle_dict, path_normalize_factor)

    def normalize(self, u):
        u_loc = self.normalize_factors["u_loc"]
        u_std = self.normalize_factors["u_std"]
        # print(u_loc, u_std)
        u_normalized = (u - u_loc) / u_std
        return u_normalized

    def add_noise(self, u):
        w = torch.randn_like(u[:, :6]) # noise
        w_b = torch.randn_like(u[0, :6])  # bias
        w[:, :3] *= self.sigma_gyro
        w[:, 3:6] *= self.sigma_acc
        w_b[:3] *= self.sigma_b_gyro
        w_b[3:6] *= self.sigma_b_acc
        u[:, :6] += w + w_b
        return u


    @staticmethod
    def read_data(args):
        raise NotImplementedError

    @classmethod
    def load(cls, *_file_name):
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(cls.pickle_extension):
            file_name += cls.pickle_extension
        with open(file_name, "rb") as file_pi:
            pickle_dict = pickle.load(file_pi)
        return pickle_dict

    @classmethod
    def dump(cls, mondict, *_file_name):
        file_name = os.path.join(*_file_name)
        if not file_name.endswith(cls.pickle_extension):
            file_name += cls.pickle_extension
        with open(file_name, "wb") as file_pi:
            pickle.dump(mondict, file_pi)

    def init_state_torch_filter(self, iekf):
        b_omega0 = torch.zeros(3).double()
        b_acc0 = torch.zeros(3).double()
        Rot_c_i0 = torch.eye(3).double()
        t_c_i0 = torch.zeros(3).double()
        return b_omega0, b_acc0, Rot_c_i0, t_c_i0  

    def get_estimates(self, dataset_name):
        #  Obtain  estimates
        dataset_name = self.datasets[dataset_name] if type(dataset_name) == int else \
            dataset_name
        file_name = os.path.join(self.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            return
        mondict = self.load(file_name)
        Rot = mondict['Rot']
        v = mondict['v']
        p = mondict['p']
        b_omega = mondict['b_omega']
        b_acc = mondict['b_acc']
        Rot_c_i = mondict['Rot_c_i']
        t_c_i = mondict['t_c_i']
        measurements_covs = mondict['measurements_covs']
        return Rot, v, p , b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs

class DataArgs():
    path_data_save = "can_bus"
    path_results = "results"
    path_temp = "temp"
    
    n_train = 20
    test_scene = 444 # 501, 523 are good ones
    
    epochs = 10
    seq_dim = 1000

    # training, cross-validation and test dataset
    cross_validation_sequences = []
    test_sequences = [f'scene-{test_scene:0>4}_ms_imu.json']
    continue_training = False

    # choose what to do
    read_data = 0
    train_filter = 0
    test_filter = 1
    results_filter = 1
    dataset_class = NuScenesData
    parameter_class = NuScenesParameters

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx