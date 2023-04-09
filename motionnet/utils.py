import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as im

# from torchsparse import SparseTensor
# from torchsparse.utils.quantize import sparse_quantize
# from torchsparse.utils.collate import sparse_collate
# from torch_scatter import scatter_max

# import rospy
# from visualization_msgs.msg import *
# from geometry_msgs.msg import Point32
# from std_msgs.msg import ColorRGBA


# Remove classifier layer
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def remove_classifier(seg_net):
    seg_net.classifier = Identity()
    return seg_net


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def iou_one_frame(pred, target, n_classes=23):
    # pred = pred.view(-1).detach().cpu().numpy()
    # target = target.view(-1).detach().cpu().numpy()
    # n_classes -= 1
    # intersection = np.zeros(n_classes)
    # union = np.zeros(n_classes)

    pred = pred.view(-1)
    target = target.view(-1)
    # n_classes -= 1

    uni_classes = torch.unique(target)
    n_classes = uni_classes.shape[0]

    intersection = torch.zeros(n_classes)
    union = torch.zeros(n_classes)

    # for cls in range(n_classes):
    #     cls_array = cls
    #     cls += 1
    #     intersection[cls_array] = torch.sum((pred == cls) & (target == cls))
    #     union[cls_array] = torch.sum((pred == cls) | (target == cls))
    for cls in range(n_classes):
        class_t = uni_classes[cls]
        intersection[cls] = torch.sum((pred == class_t) & (target == class_t))
        union[cls] = torch.sum((pred == class_t) | (target == class_t))
    return intersection, union, uni_classes


def generate_seg_in(lidar, res):
    # Create input data
    coords = np.round(lidar[:, :3] / res)
    coords -= coords.min(0, keepdims=1)
    feats = lidar
    # Filter out duplicate points
    coords, indices, inverse = sparse_quantize(coords, return_index=True, return_inverse=True)
    coords = torch.tensor(coords, dtype=torch.int)
    feats = torch.tensor(feats[indices], dtype=torch.float)

    inputs = SparseTensor(coords=coords, feats=feats)
    inputs = sparse_collate([inputs]).cuda()
    return inputs, inverse


def points_to_voxels(lidar, point_features, carla_ds):
    # Mask inliers
    points = lidar[:, :3]
    valid_point_mask = np.all((points < carla_ds.max_bound) & (points >= carla_ds.min_bound), axis=1)
    point_features = point_features[valid_point_mask, :]
    points = points[valid_point_mask, :]
    # Convert to voxels
    voxels = np.floor((points - carla_ds.min_bound) / carla_ds.voxel_sizes).astype(np.int)
    # Clamp to account for any floating point errors
    maxes = np.reshape(carla_ds.grid_dims - 1, (1, 3))
    mins = np.zeros_like(maxes)
    voxels = np.clip(voxels, mins, maxes).astype(np.int)
    return voxels, point_features


def transform_lidar(points, relative_pose):
    points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]
    return points


def create_bev(lidar, point_features, carla_ds, data_params, device):
    voxels, point_features = points_to_voxels(lidar, point_features, carla_ds)
    # Get highest xy
    flat_xy = voxels[:, :2]
    height = voxels[:, 2] + 1000
    feat_index = data_params["x_dim"] * flat_xy[:, 1] + flat_xy[:, 0]
    feat_index = torch.from_numpy(feat_index).to(device).long()
    height = torch.from_numpy(height).to(device)
    flat_highest_z = torch.zeros(data_params["x_dim"] * data_params["y_dim"], device=device).long()
    flat_highest_z, argmax_flat_spatial_map = scatter_max(
        height,
        feat_index,
        dim=0,
        out=flat_highest_z
    )
    highest_indices = argmax_flat_spatial_map[argmax_flat_spatial_map != point_features.shape[0]]
    # Create input
    highest_feats = point_features[highest_indices, :]
    voxels = torch.from_numpy(voxels).long().to(device)
    highest_x = voxels[highest_indices, 0]
    highest_y = voxels[highest_indices, 1]
    bev_features = torch.zeros((data_params["x_dim"], data_params["y_dim"], point_features.shape[1]), device=device)
    bev_features[highest_x, highest_y, :] = highest_feats
    proj_indices = torch.zeros((data_params["x_dim"], data_params["y_dim"]), device=device).long() - 1
    proj_indices[highest_x, highest_y] = feat_index[highest_indices].long()
    mask_inliers = torch.sum(bev_features, dim=2) != 0
    return bev_features, proj_indices, mask_inliers


def generate_smnet_input(pose_batch, horizon_batch, carla_ds, n_obj_classes, seg_net, device, model_params, data_params,
                         input_feats=None):
    B = len(pose_batch)
    T = len(pose_batch[0])
    H = int(carla_ds.grid_dims[0])
    W = int(carla_ds.grid_dims[1])
    C = model_params["ego_feat_dim"]
    all_feats = torch.zeros((B, T, H, W, C), device=device)
    all_indices = torch.zeros((B, T, H, W), device=device).long() - 1
    all_inliers = torch.zeros((B, T, H, W), device=device).long()
    for b_i in range(len(pose_batch)):
        for t_i in range(len(pose_batch[b_i])):
            lidar = horizon_batch[b_i][t_i]
            if lidar.shape[0] == 1:
                continue
            else:
                with torch.no_grad():
                    if input_feats is None:
                        seg_input, inv = generate_seg_in(lidar, model_params["vres"])
                        point_features = seg_net(seg_input.to(device))[inv]
                    else:
                        point_features = torch.tensor(input_feats[b_i][t_i], device=device)
                    transformed_points = transform_lidar(lidar[:, :3], pose_batch[b_i][t_i])
                    bev_features, proj_indices, mask_inliers = \
                        create_bev(transformed_points, point_features, carla_ds, data_params, device)
                    all_feats[b_i, t_i, :, :, :] = bev_features
                    all_indices[b_i, t_i, :, :] = proj_indices.long()
                    all_inliers[b_i, t_i, :, :] = mask_inliers.long()
    return all_feats, all_indices, all_inliers


def gen_convbki_input(pose_batch, horizon_batch, seg_net, device, model_params):
    all_labels = []
    all_points = []
    for b_i in range(len(pose_batch)):
        labels_b = []
        points_b = []
        for t_i in range(len(pose_batch[b_i])):
            lidar = horizon_batch[b_i][t_i]
            if lidar.shape[0] == 1:
                continue
            else:
                with torch.no_grad():
                    seg_input, inv = generate_seg_in(lidar, model_params["vres"])
                    point_labels = F.softmax(seg_net(seg_input.to(device))[inv], dim=1)
                    transformed_points = transform_lidar(lidar[:, :3], pose_batch[b_i][t_i])
                    labels_b.append(point_labels)
                    points_b.append(transformed_points)
        all_labels.append(labels_b)
        all_points.append(points_b)
    return all_points, all_labels


def points_to_voxels_torch(voxel_grid, points, min_bound, grid_dims, voxel_sizes):
    voxels = torch.floor((points - min_bound) / voxel_sizes).to(dtype=torch.int)
    # Clamp to account for any floating point errors
    maxes = (grid_dims - 1).reshape(1, 3)
    mins = torch.zeros_like(maxes)
    voxels = torch.clip(voxels, mins, maxes).to(dtype=torch.long)

    voxel_grid = voxel_grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    return voxel_grid


def remap_colors(colors):
    # color
    colors_temp = np.zeros((len(colors), 3))
    for i in range(len(colors)):
        colors_temp[i, :] = colors[i]
    colors = colors_temp.astype("int")
    colors = colors / 255.0
    return colors


def publish_local_map(labeled_grid, centroids, max_dim, min_dim, grid_dims, colors, next_map):
    next_map.markers.clear()
    marker = Marker()
    marker.id = 0
    marker.ns = "Local Semantic Map"
    marker.header.frame_id = "map"
    marker.type = marker.CUBE_LIST
    marker.action = marker.ADD
    marker.lifetime.secs = 0
    marker.header.stamp = rospy.Time.now()

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1

    marker.scale.x = (max_dim[0] - min_dim[0]) / grid_dims[0]
    marker.scale.y = (max_dim[1] - min_dim[1]) / grid_dims[1]
    marker.scale.z = (max_dim[2] - min_dim[2]) / grid_dims[2]

    X, Y, Z, C = labeled_grid.shape
    semantic_labels = labeled_grid.view(-1, C).detach().cpu().numpy()
    centroids = centroids.detach().cpu().numpy()

    semantic_sums = np.sum(semantic_labels, axis=-1, keepdims=False)
    valid_mask = semantic_sums >= 1

    semantic_labels = semantic_labels[valid_mask, :]
    centroids = centroids[valid_mask, :]

    semantic_labels = np.argmax(semantic_labels / np.sum(semantic_labels, axis=-1, keepdims=True), axis=-1)
    semantic_labels = semantic_labels.reshape(-1, 1)

    for i in range(semantic_labels.shape[0]):
        pred = semantic_labels[i]
        point = Point32()
        color = ColorRGBA()
        point.x = centroids[i, 0]
        point.y = centroids[i, 1]
        point.z = centroids[i, 2]
        color.r, color.g, color.b = colors[pred].squeeze()

        color.a = 1.0
        marker.points.append(point)
        marker.colors.append(color)

    next_map.markers.append(marker)
    return next_map

# def get_batched(dataset, batch_size = 4):
#     total_len = len(dataset) - 1
#     list_data = np.arange(total_len)
#     np.random.shuffle(list_data)

#     batch = []
#     for i in range (int(len(dataset)/4)):            
#         b = i * batch_size
#         input_data_b = []
#         output_b = []
#         for j in range(batch_size):
#             input_data_t, output_t = dataset[list_data[b+j]]
#             input_data_b.append(input_data_t)
#             output_b.append(output_t)

#         batch.append([input_data_b, output_b])

#     return batch


def get_batch(dataset, list_data, i, batch_size = 4):

    input_data_batch = []
    output_batch = []
    b = i * batch_size
    for j in range(batch_size):
        input_data, output = dataset[list_data[b+j]]
        input_data_batch.append(input_data)
        output_batch.append(output)
    return input_data_batch, output_batch

# Form BEV
def form_bev(labels, grid_dims):
    bev_map = np.zeros((grid_dims[0], grid_dims[1]))

    bev_mask = np.sum(labels,axis=-1) != 0
    bev_x,bev_y = np.where(bev_mask)

    for i in range(bev_x.shape[0]):
        x = bev_x[i]
        y = bev_y[i]

        lables_column = labels[x,y]
        
        mask_zero = lables_column != 0
        mask = np.sum(mask_zero) >= 1
        
        if mask:
            indx = np.where(mask_zero)[0][-1]
            bev_map[x, y] = labels[x,y,indx]

            
    return bev_map.astype(np.uint8)

    
def bev_img(bev_map):
    colored_data = LABEL_COLORS[bev_map.astype(int)]
    img = im.fromarray(colored_data, 'RGB')
    return img

LABEL_COLORS = np.array([
    (  255,   255,   255,), # unlabled
    (245, 150, 100,), #car
    (245, 230, 100,), #bike
    ( 30,  30, 255,), #person
    (255,   0, 255,), #road
    (255, 150, 255,), #parking
    ( 75,   0,  75,), #sidewalk
    (  0, 200, 255,), #building
    ( 50, 120, 255,), #fence
    (  0, 175,   0,), #vegetation
    ( 80, 240, 150,), #terrain
    (150, 240, 255,), #pole
    (90,  30, 150,), #traffic-sign
    (255,  0,  0,), #moving-car
    ( 0,  0, 255,), #moving-person
]).astype(np.uint8)