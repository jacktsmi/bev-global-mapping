import os
import shutil
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict
from dataset import BaseDataset
from utils_torch_filter import TORCHIEKF
from utils_numpy_filter import NUMPYIEKF as IEKF
from utils import prepare_data
from train_torch_filter import train_filter
from utils_plot import results_filter
from nuscenes_data import NuScenesData, DataArgs


def launch(args):
    if args.read_data:
        args.dataset_class.read_data(args)
    dataset = args.dataset_class(args)

    if args.train_filter:
        train_filter(args, dataset)

    if args.test_filter:
        test_filter(args, dataset)

    if args.results_filter:
        results_filter(args, dataset)


class KITTIParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))

def test_filter(args, dataset):
    iekf = IEKF()
    torch_iekf = TORCHIEKF()

    # put Kitti parameters
    iekf.filter_parameters = KITTIParameters()
    iekf.set_param_attr()
    torch_iekf.filter_parameters = KITTIParameters()
    torch_iekf.set_param_attr()

    torch_iekf.load(args, dataset)
    iekf.set_learned_covariance(torch_iekf)

    for i in range(0, len(args.test_sequences)):
        dataset_name = args.test_sequences[i]
        print("Test filter on sequence: " + dataset_name)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, i,
                                                       to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double()
        measurements_covs = torch_iekf.forward_nets(u_t)
        measurements_covs = measurements_covs.detach().cpu().numpy()
        start_time = time.time()
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i = iekf.run(t, u, measurements_covs,
                                                                   v_gt, p_gt, N,
                                                                   ang_gt[0])
        diff_time = time.time() - start_time
        print("Execution time: {:.2f} s (sequence time: {:.2f} s)".format(diff_time,
                                                                          t[-1] - t[0]))
        mondict = {
            't': t, 'Rot': Rot, 'v': v, 'p': p, 'b_omega': b_omega, 'b_acc': b_acc,
            'Rot_c_i': Rot_c_i, 't_c_i': t_c_i,
            'measurements_covs': measurements_covs,
            }
        dataset.dump(mondict, args.path_results, dataset_name + "_filter.p")

if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    args = DataArgs()
    dataset = NuScenesData(args)
    launch(DataArgs)
