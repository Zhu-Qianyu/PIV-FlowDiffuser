import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import metrics
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datasets

from torch.utils.data import Dataset
from piv_flowdiffuser import FlowDiffuser


from torchvision import transforms
class DatasetPIV(Dataset):
    def __init__(self, data_dir, matching=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        data_list = os.listdir(self.data_dir)

        if matching is None:
            self.data_list = data_list
        else:
            self.data_list = [s for s in data_list if matching in s]
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_list[idx])
        data = np.load(data_path)
        img1 = data['img1']
        img2 = data['img2']
        # Assuming img1 and img2 are shape (H, W) for grayscale, or (H, W, C) for RGB
        # Convert to RGB if they are grayscale
        if img1.ndim == 2:
            img1 = np.stack((img1,)*3, axis=-1)  # Convert to (H, W, 3)
        if img2.ndim == 2:
            img2 = np.stack((img2,)*3, axis=-1)  # Convert to (H, W, 3)
        u = data['u']
        v = data['v']
        flow = np.stack((u, v), axis=-1)
        valid = np.ones_like(u)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            flow = self.transform(flow)
            valid = self.transform(valid)
        return img1, img2, flow, valid

def val_CAI(model, args, iters=12 ):
    model.eval()

    to_tensor = transforms.Compose([
    transforms.ToTensor()  # Converts a PIL Image or numpy.ndarray to tensor.
])
    cai_test_path = '/media/newdisk/datasets_piv/piv_cai/test'
    cai_test_data = DatasetPIV(cai_test_path, transform=to_tensor)

    val_dataset = cai_test_data

    print(len(val_dataset))
    epe = []
    rmse = []
    aae = []
    aepe = []
    for i in range(len(val_dataset)):
        img1, img2, flow, valid_gt = val_dataset[i]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()
        # print(img1.shape)
        flow_low, flow_prediction = model(img1, img2, iters=iters, test_mode=True)
        flow = flow.squeeze(0)
        flow_prediction = flow_prediction[-1].squeeze(0)

        u_gt = flow[1].detach().cpu()
        v_gt = flow[0].detach().cpu()
        if args.modty == "1":
            u_pd = flow_prediction[1].detach().cpu()
            v_pd = flow_prediction[0].detach().cpu()
        else:
            u_pd = flow_prediction[0].detach().cpu()
            v_pd = flow_prediction[1].detach().cpu()

        eepe = metrics.EPE(u_gt, v_gt, u_pd, v_pd)
        eepe = np.sum(np.array(eepe))

        ermse = metrics.RMSE(u_gt, v_gt, u_pd, v_pd)
        eaae = metrics.AAE(u_gt, v_gt, u_pd, v_pd)
        eaepe = metrics.AEE(u_gt, v_gt, u_pd, v_pd)

        epe.append(eepe)
        rmse.append(ermse)
        aae.append(eaae)
        aepe.append(eaepe)


    all_epe_data = np.concatenate([epe_data.flatten() for epe_data in epe])
    all_aae_data = np.concatenate([aae_data.flatten() for aae_data in aae])
    all_rmse_data = np.concatenate([rmse_data.flatten() for rmse_data in rmse])
    all_aepe_data = np.concatenate([aepe_data.flatten() for aepe_data in aepe])

    epe = np.average(all_epe_data)
    aae = np.average(all_aae_data)
    rmse = np.average(all_rmse_data)
    aepe = np.average(all_aepe_data)

    print("epe = {}".format(epe))
    print("aee = {}".format(aae))
    print("rmse = {}".format(rmse))
    print("aepe = {}".format(aepe))

def val_P2(model, args, iters=12 ):
    model.eval()

    to_tensor = transforms.Compose([
    transforms.ToTensor()  # Converts a PIL Image or numpy.ndarray to tensor.
])

    raft256_valid_path = '/media/newdisk/datasets_piv/piv_raft/Data_ProblemClass2_RAFT-PIV/Validation_Dataset_ProblemClass2_RAFT256-PIV'
    raft256_valid_data =  DatasetPIV(raft256_valid_path,transform=to_tensor)
    val_dataset = raft256_valid_data

    print(len(val_dataset))
    epe = []
    rmse = []
    aae = []
    aepe = []
    for i in range(len(val_dataset)):
        img1, img2, flow, valid_gt = val_dataset[i]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()

        flow_low, flow_prediction = model(img1, img2, iters=iters, test_mode=True)
        flow = flow.squeeze(0)
        flow_prediction = flow_prediction[-1].squeeze(0)

        u_gt = flow[0].detach().cpu()
        v_gt = flow[1].detach().cpu()
        if args.modty == "2":
            u_pd = flow_prediction[0].detach().cpu()
            v_pd = flow_prediction[1].detach().cpu()

        elif args.modty == "1" :
            u_pd = flow_prediction[1].detach().cpu()
            v_pd = flow_prediction[0].detach().cpu()

        eepe = metrics.EPE(u_gt, v_gt, u_pd, v_pd)
        eepe = np.sum(np.array(eepe))

        ermse = metrics.RMSE(u_gt, v_gt, u_pd, v_pd)
        eaae = metrics.AAE(u_gt, v_gt, u_pd, v_pd)
        eaepe = metrics.AEE(u_gt, v_gt, u_pd, v_pd)

        epe.append(eepe)
        rmse.append(ermse)
        aae.append(eaae)
        aepe.append(eaepe)


    all_epe_data = np.concatenate([epe_data.flatten() for epe_data in epe])
    all_aae_data = np.concatenate([aae_data.flatten() for aae_data in aae])
    all_rmse_data = np.concatenate([rmse_data.flatten() for rmse_data in rmse])

    epe = np.average(all_epe_data)
    aae = np.average(all_aae_data)
    rmse = np.average(all_rmse_data)
    aepe = np.average(aepe)

    print("epe = {}".format(epe))
    print("aae = {}".format(aae))
    print("rmse = {}".format(rmse))
    print("aepe = {}".format(aepe))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--modty', help="restore checkpoint type")
    args = parser.parse_args()

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()
    
    with torch.no_grad():
        if  args.dataset == 'CAI':
            val_CAI(model.module)
        if args.dataset == 'P2':
            val_P2(model.module)