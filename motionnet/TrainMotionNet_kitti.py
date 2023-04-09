import sys

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
from Data.kitti_dataset import *
from MotionNet.Model import *
from utils import *


from tqdm import tqdm

MODEL_CONFIG = "MotionNet"
DATA_CONFIG = "kitti"

DEBUG = True

model_params_file = os.path.join(os.getcwd(), "Configs", MODEL_CONFIG + ".yaml")
print(model_params_file)
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

data_params_file = os.path.join(os.getcwd(), "Configs", DATA_CONFIG + ".yaml")
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = model_params["T"]
past_frames = model_params["past_frames"]
REMOVE_ZERO = model_params["remove_zero"]
train_dir = data_params["train_dir"]
val_dir = data_params["val_dir"]
test_dir = data_params["test_dir"]
num_workers = data_params["num_workers"]
remap = data_params["remap"]
voxelize_input = model_params["voxelize_input"]
binary_counts = model_params["binary_counts"]
transform_pose = model_params["transform_pose"]
seed = model_params["seed"]
B = model_params["B"]
BETA1 = model_params["BETA1"]
BETA2 = model_params["BETA2"]
decayRate = model_params["DECAY"]
lr = model_params["lr"]
epoch_num = model_params["epoch_num"]

coor_ranges = data_params['min_bound'] + data_params['max_bound']
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / data_params["x_dim"],
              abs(coor_ranges[4] - coor_ranges[1]) / data_params["y_dim"],
              abs(coor_ranges[5] - coor_ranges[2]) / data_params["z_dim"]]

model = MotionNet(height_feat_size=data_params["z_dim"], num_classes=data_params["num_classes"], T=T).to(device)


criterion = nn.CrossEntropyLoss()
("loading dataset to memeory...")
grid_size_fromyaml =[data_params["x_dim"], data_params["y_dim"], data_params["z_dim"]]

grid_size_fromyaml2 =[data_params["x_dim2"], data_params["y_dim2"], data_params["z_dim2"]]

carla_ds = KittiDataset(directory=train_dir, device=device, num_frames=T, past_frames=past_frames, transform_pose=transform_pose, split="train", grid_size=grid_size_fromyaml, grid_size_train=grid_size_fromyaml2)
dataloader_train = DataLoader(carla_ds, batch_size=B, shuffle=True, collate_fn=carla_ds.collate_fn, num_workers=num_workers, pin_memory=True)

val_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, past_frames=past_frames, transform_pose=transform_pose, split="valid", grid_size=grid_size_fromyaml, grid_size_train=grid_size_fromyaml2)
dataloader_val = DataLoader(val_ds, batch_size=B, shuffle=True, collate_fn=val_ds.collate_fn, num_workers=num_workers, pin_memory=True)

writer = SummaryWriter("./Models/Runs/" + MODEL_CONFIG)
save_dir = "./Models/Weights/" + MODEL_CONFIG

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if device == "cuda":
    torch.cuda.empty_cache()

setup_seed(seed)

optimizer = optim.Adam(model.parameters(), lr=lr)
my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.5)

train_count = 0

if DEBUG:
    save_temp_dir = "./bev_during_training"
    save_temp_dir_train_f = os.path.join(save_temp_dir, "train")
    save_temp_dir_val_f = os.path.join(save_temp_dir, "val")
    if not os.path.exists(save_temp_dir):
        os.makedirs(save_temp_dir)
        os.makedirs(save_temp_dir_train_f)
        os.makedirs(save_temp_dir_val_f)


for epoch in range(epoch_num):
    print("Current Epoch:{}, Current LR:{}".format(epoch, my_lr_scheduler.get_last_lr()))
    
    # Training
    model.train()
    count = 0
    save_temp_dir_train = os.path.join(save_temp_dir_train_f, "Epoch_" + str(epoch))
    if not os.path.exists(save_temp_dir_train):
        os.makedirs(save_temp_dir_train)

    
    for input_data, output in tqdm(dataloader_train):

        optimizer.zero_grad()

        input_data = torch.tensor(np.array(input_data)).to(device)
        output = torch.tensor(np.array(output)).to(device)
        # print("input:", input_data.shape)
        # print("output:", output.shape)
        preds = model(input_data) # B, C, H, W
        preds = torch.permute(preds, (0, 2, 3, 1))[:,28:228,28:228,:]
        # print("preds:", preds.shape)

        if DEBUG and count%200 == 0:
            save_temp_dir_frame_output = os.path.join(save_temp_dir_train, "epoch" + str(epoch) + "_" + str(count) +"_Output" + ".png")
            save_temp_dir_frame_gt = os.path.join(save_temp_dir_train, "epoch" + str(epoch) + "_" + str(count) +"_GT" + ".png")

            B, H, W, C = preds.shape
            probs_debug = torch.nn.functional.softmax(preds[-1].unsqueeze(dim=0), dim=3).view(H, W, C)
            preds_np = probs_debug.detach().cpu().numpy().astype(np.float32)
            preds_np = np.argmax(preds_np, axis=2)
            img = bev_img(preds_np)

            img_output = bev_img(output[-1].detach().cpu().numpy().astype(np.uint8))

            img.save(save_temp_dir_frame_output)
            img_output.save(save_temp_dir_frame_gt)

            for i in range(T):
                save_temp_dir_frame_input = os.path.join(save_temp_dir_train, "epoch" + str(epoch) + "_" + str(count) +"_Input_T" + str(i) + ".png")

                input_temp = input_data[-1,i,28:228,28:228,:].detach().cpu().numpy().astype(np.uint8) # Get 200 x 200 instead  256 x 256 grid
                input_bev = form_bev(input_temp, grid_size_fromyaml[:2])
                img_bev = bev_img(input_bev)
                img_bev.save(save_temp_dir_frame_input)


        count += 1
        output = output.view(-1).long()
        preds = preds.contiguous().view(-1, preds.shape[3])

        # Remove zero
        if REMOVE_ZERO:
            mask = output != 0
            output_masked = output[mask]
            preds_masked = preds[mask]
        else:
            output_masked = output
            preds_masked = preds

        loss = criterion(preds_masked, output_masked)
        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            probs = torch.nn.functional.softmax(preds_masked, dim=1)
            preds_masked = torch.argmax(probs, dim=1)
            accuracy = torch.sum(preds_masked == output_masked) / output_masked.shape[0]

            if DEBUG and count%200 == 0:
                intersection, union, uni_classes = iou_one_frame(preds_masked, output_masked,
                                                n_classes=data_params["num_classes"])

                writer.add_scalar(MODEL_CONFIG + '/mIoU/Train', torch.mean(intersection / union).detach().cpu().numpy(), train_count)


        # Record
        writer.add_scalar(MODEL_CONFIG + '/Loss/Train', loss.item(), train_count)
        writer.add_scalar(MODEL_CONFIG + '/Accuracy/Train', accuracy, train_count)

        train_count += input_data.shape[0]

    # Save model, decreaser learning rate
    my_lr_scheduler.step()
    torch.save(model.state_dict(), os.path.join(save_dir, "Epoch" + str(epoch) + ".pt"))

    # Validation
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        counter = 0
        num_correct = 0
        num_total = 0
        all_intersections = torch.zeros(data_params["num_classes"] - 1)
        all_unions = torch.zeros(data_params["num_classes"] - 1) + 1e-6  # SMOOTHING
        count = 0
        save_temp_dir_val = os.path.join(save_temp_dir_val_f, "Epoch_" + str(epoch))
        if not os.path.exists(save_temp_dir_val):
            os.makedirs(save_temp_dir_val)

        for input_data, output in tqdm(dataloader_val):
            optimizer.zero_grad()
            input_data = torch.tensor(np.array(input_data)).to(device)
            output = torch.tensor(np.array(output)).to(device)
            preds = model(input_data)
            preds = torch.permute(preds, (0, 2, 3, 1))[:,28:228,28:228,:]

            if DEBUG and count%100 == 0:
                save_temp_dir_frame_output = os.path.join(save_temp_dir_val, "epoch" + str(epoch) + "_" + str(count) +"_Output" + ".png")
                save_temp_dir_frame_gt = os.path.join(save_temp_dir_val, "epoch" + str(epoch) + "_" + str(count) +"_GT" + ".png")
                # save_temp_dir_frame_input = os.path.join(save_temp_dir_val, "epoch" + str(epoch) + "_" + str(count) +"_Input" + ".png")

                B, H, W, C = preds.shape
                probs_debug = torch.nn.functional.softmax(preds[-1].unsqueeze(dim=0), dim=3).view(H, W, C)
                preds_np = probs_debug.detach().cpu().numpy().astype(np.float32)
                preds_np = np.argmax(preds_np, axis=2)
                img = bev_img(preds_np)

                img_output = bev_img(output[-1].detach().cpu().numpy().astype(np.uint8))

                # input_temp = input_data[0,-1,28:228,28:228,:].detach().cpu().numpy().astype(np.uint8)
                # input_bev = form_bev(input_temp, grid_size_fromyaml[:2])
                # img_bev = bev_img(input_bev)

                img.save(save_temp_dir_frame_output)
                img_output.save(save_temp_dir_frame_gt)
                # img_bev.save(save_temp_dir_frame_input)

                for i in range(T):
                    save_temp_dir_frame_input = os.path.join(save_temp_dir_train, "epoch" + str(epoch) + "_" + str(count) +"_Input_T" + str(i) + ".png")

                    input_temp = input_data[-1,i,28:228,28:228,:].detach().cpu().numpy().astype(np.uint8) # Get 200 x 200 instead  256 x 256 grid
                    input_bev = form_bev(input_temp, grid_size_fromyaml[:2])
                    img_bev = bev_img(input_bev)
                    img_bev.save(save_temp_dir_frame_input)
            
            count += 1
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[3])

            # Remove zero
            if REMOVE_ZERO:
                mask = output != 0
                output_masked = output[mask]
                preds_masked = preds[mask]
            else:
                output_masked = output
                preds_masked = preds

            loss = criterion(preds_masked, output_masked)

            running_loss += loss.item()
            counter += input_data.shape[0]

            # Accuracy
            probs = torch.nn.functional.softmax(preds_masked, dim=1)
            preds_masked = torch.argmax(probs, dim=1)
            num_correct += torch.sum(preds_masked == output_masked)
            num_total += output_masked.shape[0]

            intersection, union, uni_classes = iou_one_frame(preds_masked, output_masked,
                                                n_classes=data_params["num_classes"])

            for cls in range(uni_classes.shape[0]):
                class_t = uni_classes[cls] - 1
                all_intersections[class_t] += intersection[cls]
                all_unions[class_t] += union[cls]

        print(f'Eppoch Num: {epoch} ------ average val loss: {running_loss / counter}')
        print(f'Eppoch Num: {epoch} ------ average val accuracy: {num_correct / num_total}')
        print(f'Eppoch Num: {epoch} ------ val miou: {torch.mean(all_intersections / all_unions).detach().cpu().numpy()}')
        writer.add_scalar(MODEL_CONFIG + '/Loss/Val', running_loss / counter, epoch)
        writer.add_scalar(MODEL_CONFIG + '/Accuracy/Val', num_correct / num_total, epoch)
        writer.add_scalar(MODEL_CONFIG + '/mIoU/Val', torch.mean(all_intersections / all_unions).detach().cpu().numpy(), epoch)

writer.close()