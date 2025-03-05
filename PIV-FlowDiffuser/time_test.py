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
import time
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

def val_CAI(model, iters=12 ):
    model.eval()

    to_tensor = transforms.Compose([
    transforms.ToTensor()  # Converts a PIL Image or numpy.ndarray to tensor.
])
    cai_test_path = '/media/newdisk/datasets_piv/piv_cai/test'
    cai_test_data = DatasetPIV(cai_test_path, transform=to_tensor)
    val_dataset = cai_test_data

    # P2_test_path = '/media/newdisk/datasets_piv/piv_raft/Data_ProblemClass2_RAFT-PIV/Validation_Dataset_ProblemClass2_RAFT256-PIV'
    # P2_test_data = DatasetPIV(P2_test_path, transform=to_tensor)
    # val_dataset = P2_test_data

    start_time = time.time()
    for i in range(450):
        img1, img2, flow, valid_gt = val_dataset[i]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()
 
        flow_low, flow_prediction = model(img1, img2, iters=iters, test_mode=True)
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total evaluation time: {total_time:.8f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    # parser.add_argument('--savedir', help="save dir")
    args = parser.parse_args()

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if  args.dataset == 'CAI':
            val_CAI(model.module)