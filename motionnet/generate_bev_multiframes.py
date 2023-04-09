import numpy as np
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image as im
from tqdm import tqdm
from ShapeContainer import ShapeContainer


PAST_FRAME = 50 #inclduing current
FUTURE_FRAME = 50

grid_dims = np.array([256,256,64])
min_bound=np.array([-25.6, -25.6, -6.4], dtype=np.float32)
max_bound=np.array([25.6, 25.6,  6.4], dtype=np.float32)
config_file = os.path.join('Data/semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))

LABELS_REMAP_TEMP = kitti_config["LABELS_REMAP_TEMP"]
LABELS_REMAP = kitti_config["learning_map_temp"]

SAVE_VOXELS = False

def find_horizon(idx, PAST_FRAME=50, FUTURE_FRAME=50, TOTAL=100):
    
    idx_past = np.arange(idx-PAST_FRAME, idx) + 1
    idx_past_valid = np.where(idx_past >= 0)
    idx_past = idx_past[idx_past_valid]

    idx_future = np.arange(idx, idx+FUTURE_FRAME) + 1
    idx_future_valid = np.where(idx_future < TOTAL)
    idx_future = idx_future[idx_future_valid]

    idx_total = np.hstack((idx_past, idx_future))

    return(idx_total)


def initialize_grid(grid_size=np.array([256., 256., 32.]),
        min_bound=np.array([-25.6, -25.6, -2], dtype=np.float32),
        max_bound=np.array([25.6, 25.6,  4.4], dtype=np.float32),
        num_channels=15,
        coordinates="cartesian"):
        
    cart_mat = ShapeContainer(grid_size, min_bound, max_bound, num_channels, coordinates)
    return cart_mat

# Add points to grid
def add_points(points, labels, grid):
    pointlabels = np.hstack((points, labels))
    grid[pointlabels] = grid[pointlabels] + 1
    return grid

def get_pose(idx, poses, Tr_s):
    pose = np.zeros((4, 4))
    pose[3, 3] = 1
    pose[:3, :4] = poses[idx,:].reshape(3, 4)

    Tr = np.zeros((4, 4))
    Tr[3, 3] = 1
    Tr[:3, :4] = Tr_s[idx,:].reshape(3,4)

    Tr = Tr.astype(np.float32)
    pose = pose.astype(np.float32)
    global_pose = np.matmul(np.linalg.inv(Tr), np.matmul(pose, Tr))

    return global_pose

# Form BEV
def form_bev(labels):
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


sequence_path = "/mnt/data/kitti/sequences"

for scene in sorted(os.listdir(sequence_path)):
    if int(scene) != 0:
        continue
    # if int(scene) > 10:
    #     break
    print(scene)
    scene_path = os.path.join(sequence_path, scene) 

    points_path = os.path.join(scene_path, 'velodyne')
    labels_path = os.path.join(scene_path, 'labels')
    poses_path = os.path.join(scene_path, "poses.txt")
    tr_path = os.path.join(scene_path, "calib.txt")
    bev_gt_path = os.path.join(scene_path, 'bev_gt2')
    bev_img_path = os.path.join(scene_path, 'bev_img2')


    if not os.path.exists(bev_gt_path):
        os.makedirs(bev_gt_path)
    if not os.path.exists(bev_img_path):
        os.makedirs(bev_img_path)    

    if SAVE_VOXELS:
        voxel_path = os.path.join(scene_path, 'voxel_new')
        if not os.path.exists(voxel_path):
            os.makedirs(voxel_path) 

    poses = np.loadtxt(poses_path)
    Tr = np.genfromtxt(tr_path)[-1,1:]
    Tr = np.repeat(np.expand_dims(Tr, axis=1).T,poses.shape[0],axis=0)
    poses_transformed = []
    # Store all labels and points in memeory
    labels_total_list = []
    points_total_list = []
    print("loading to memeory...")
    for i in tqdm(range(len(os.listdir(labels_path)))):
        labels_file = os.path.join(labels_path, str(i).zfill(6) + ".label")
        points_file = os.path.join(points_path, str(i).zfill(6) + ".bin")
        labels = np.fromfile(labels_file,dtype=np.uint32) & 0xFFFF # Not remapped
        for k in range(labels.shape[0]):
            labels[k] = LABELS_REMAP_TEMP[labels[k]] # remaped
        labels = labels.astype(np.uint8)
        points = np.fromfile(points_file,dtype=np.float32).reshape(-1,4)[:,:3]
        labels_total_list.append(labels)
        points_total_list.append(points)
        poses_transformed.append(get_pose(i, poses, Tr))

    # Initialize grid
    voxel_grid = initialize_grid(grid_size=grid_dims, max_bound=max_bound, min_bound=min_bound)
    print("processing...")
    for i in tqdm(range(len(os.listdir(labels_path)))):
        # Rest grid
        voxel_grid.reset_grid()

        idx = find_horizon(i, PAST_FRAME, FUTURE_FRAME, len(os.listdir(labels_path)))
        # ego_pose = get_pose(i, poses, Tr)
        ego_pose = poses_transformed[i]
        to_ego = np.linalg.inv(ego_pose)
        
        for j in range(idx.shape[0]):
            labels = labels_total_list[idx[j]]
            points = points_total_list[idx[j]]

            mask = labels!= 0 
            labels = labels[mask]
            points = points[mask]
            labels = np.expand_dims(labels, axis=1)

            # to_world = get_pose(idx[j], poses, Tr)
            to_world = poses_transformed[idx[j]]
            relative_pose = np.matmul(to_ego, to_world)
            transformed_points = np.dot(relative_pose[:3, :3], points.T).T + relative_pose[:3, 3]     

            voxel_grid = add_points(transformed_points, labels, voxel_grid)

        voxels = voxel_grid.get_voxels()
        labels = np.argmax(voxels, axis=3).astype(np.uint8)
        bev_map = form_bev(labels)
        img = bev_img(bev_map)

        if SAVE_VOXELS:
            voxel_file = os.path.join(voxel_path, str(i).zfill(6) + ".label")
            labels.tofile(voxel_file)

        bev_file = os.path.join(bev_gt_path, str(i).zfill(6) + ".bin")
        bev_map.tofile(bev_file)

        bev_img_file = os.path.join(bev_img_path, str(i).zfill(6) + ".png")
        img.save(bev_img_file)

        #img.show()


        