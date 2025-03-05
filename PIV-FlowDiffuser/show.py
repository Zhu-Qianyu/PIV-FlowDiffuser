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

def plot_field(x,y,u,v,bkg=None,cmap=None,figsize=(8,6)):
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # 确保文本以可编辑形式保存
    assert len(x.shape) == 2, "the 2D data is required"
    def auto_step(x):
        sz = x.shape
        dx,dy=sz[0]//33, sz[1]//33
        return dx,dy
    
    fig=plt.figure(figsize=figsize)
    if bkg is not None:
        plt.imshow(bkg, cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(x*0+1,cmap="gray",vmax=1.0,vmin=0.0)

    dx,dy = auto_step(x)
    plt.quiver(y[::dx, ::dy], x[::dx, ::dy], v[::dx, ::dy], -u[::dx, ::dy])
    plt.axis('off')
    return fig

def plot_field_contrast(u, v, u_pd, v_pd, bkg=None, cmap=None, size=256, figsize=(8, 8)):  # 适当增大画布高度，方便放置色条
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  # 确保文本以可编辑形式保存
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")

    def auto_step(x):
        sz = x.shape
        dx, dy = sz[0] // 33, sz[1] // 33
        return dx, dy

    amp1 = np.sqrt(u ** 2 + v ** 2)
    amp2 = np.sqrt(u_pd ** 2 + v_pd ** 2)
    vmin_amp = min(amp1.min(), amp2.min())
    vmax_amp = max(amp1.max(), amp2.max())

    if bkg is not None:
        plt.imshow(bkg, cmap=cmap)
        plt.colorbar()
    else:
        plt.imshow(x * 0 + 1, cmap="gray", vmax=1.0, vmin=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=figsize)
    dx, dy = auto_step(x)
    ax1.quiver(y[::dx, ::dy], x[::dx, ::dy], v[::dx, ::dy], -u[::dx, ::dy])
    ax2.quiver(y[::dx, ::dy], x[::dx, ::dy], v_pd[::dx, ::dy], -u_pd[::dx, ::dy])

    im1 = ax1.imshow(amp1, cmap=cmap, vmin=vmin_amp, vmax=vmax_amp)
    im2 = ax2.imshow(amp2, cmap=cmap, vmin=vmin_amp, vmax=vmax_amp)

    # 创建用于放置色条的Axes对象，将其放置在图像下方
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # 根据需要调整位置和大小参数

    fig.colorbar(im1, cax=cax, orientation='horizontal')  # 设置色条为水平方向

    ax1.set_title("gt");
    ax2.set_title("pred")
    ax1.axis('off');
    ax2.axis('off');

    return fig


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
        # Apply transforms
        # u and v are not transformed and returned as is
        u = data['u']
        v = data['v']
        flow = np.stack((u, v), axis=-1)
        valid = np.ones_like(u)
        # Combining img1, img2 into a single tensor with shape [3, H, 2*W]
        # or you can return them separately as a tuple
        # Here, we'll return them as a tuple for simplicity
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            flow = self.transform(flow)
            valid = self.transform(valid)
        return img1, img2, flow, valid

def show_P2(model, args, iters=12):
    model.eval()

    to_tensor = transforms.Compose([
    transforms.ToTensor()  # Converts a PIL Image or numpy.ndarray to tensor.
    ])
    P2_test_path = '/media/newdisk/datasets_piv/piv_raft/Data_ProblemClass2_RAFT-PIV/Validation_Dataset_ProblemClass2_RAFT256-PIV'
    P2_test_data = DatasetPIV(P2_test_path, transform=to_tensor)
    val_dataset = P2_test_data
    savedir = args.savedir

    result_folder_pred = os.path.join(savedir, 'flow')
    epe_eval = os.path.join(savedir, 'epe')
    os.makedirs(epe_eval, exist_ok=True)
    os.makedirs(result_folder_pred, exist_ok=True)

    epe = []

    for i in range(30):
        img1, img2, flow, valid_gt = val_dataset[i]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()

        flow_low, flow_prediction = model(img1, img2, iters=iters, test_mode=True)
        flow = flow.squeeze(0)
        flow_prediction = flow_prediction[-1].squeeze(0)
        x,y = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")

        height, width = flow.shape[1:]

        u_gt = flow[0].detach().cpu()
        v_gt = flow[1].detach().cpu()

        if args.modty == "2":
            u_pd = flow_prediction[0].detach().cpu()
            v_pd = flow_prediction[1].detach().cpu()

        elif args.modty == "1" :
            u_pd = flow_prediction[1].detach().cpu()
            v_pd = flow_prediction[0].detach().cpu()

        eepe =metrics.EPE(u_gt, v_gt, u_pd, v_pd)
        epe.append(eepe)

        fig = plot_field_contrast(u_gt, v_gt, u_pd, v_pd)
        save_path = os.path.join(result_folder_pred, f"u_v_gt_pred_{i:03d}.png")
        plt.savefig(save_path)
        plt.close()

    min_value = 0
    max_value = 3
    o = 0

    epe.append(metrics.EPE(u_gt, v_gt, u_gt, v_gt))
    for eepe in epe:
        fig, ax = plt.subplots()
        im = ax.imshow(eepe, cmap='coolwarm', vmin=min_value, vmax=max_value)
        fig.colorbar(im)
        plt.title('EPE Intensity Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')

        save_path = os.path.join(epe_eval, f"epe_eval_{o:03d}.pdf")
        plt.savefig(save_path)
        plt.close()
        o += 1
def show_CAI(model, args ,iters=12):
    model.eval()

    to_tensor = transforms.Compose([
    transforms.ToTensor()  # Converts a PIL Image or numpy.ndarray to tensor.
    ])
    cai_test_path = '/media/newdisk/datasets_piv/piv_cai/test'
    cai_test_data = DatasetPIV(cai_test_path, transform=to_tensor)
    val_dataset = cai_test_data
    savedir = args.savedir

    result_folder_pred = os.path.join(savedir, 'flow')
    epe_eval = os.path.join(savedir, 'epe')
    os.makedirs(epe_eval, exist_ok=True)
    os.makedirs(result_folder_pred, exist_ok=True)

    epe = []

    for i in range(30):
        img1, img2, flow, valid_gt = val_dataset[i]
        img1 = img1[None].cuda()
        img2 = img2[None].cuda()
 
        flow_low, flow_prediction = model(img1, img2, iters=iters, test_mode=True)
        flow = flow.squeeze(0)
        flow_prediction = flow_prediction[-1].squeeze(0)
        x,y = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")

        height, width = flow.shape[1:]

        u_gt = flow[1].detach().cpu()
        v_gt = flow[0].detach().cpu()
        if args.modty == "1":
            u_pd = flow_prediction[1].detach().cpu()
            v_pd = flow_prediction[0].detach().cpu()
        else:
            u_pd = flow_prediction[0].detach().cpu()
            v_pd = flow_prediction[1].detach().cpu()

        eepe =metrics.EPE(u_gt, v_gt, u_pd, v_pd)
        epe.append(eepe)

        fig = plot_field_contrast(u_gt, v_gt, u_pd, v_pd)
        save_path = os.path.join(result_folder_pred, f"u_v_gt_pred_{i:03d}.png")
        plt.savefig(save_path)
        plt.close()

    min_value = 0
    max_value = 0.8
    o = 0

    for eepe in epe:
        fig, ax = plt.subplots()
        im = ax.imshow(eepe, cmap='coolwarm', vmin=min_value, vmax=max_value)
        fig.colorbar(im)
        plt.title('EPE Intensity Visualization')
        plt.xlabel('X')
        plt.ylabel('Y')

        save_path = os.path.join(epe_eval, f"epe_eval_{o:03d}.pdf")
        plt.savefig(save_path)
        plt.close()
        o += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--modty', help="restore checkpoint type")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--savedir', help="save dir")
    args = parser.parse_args()

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42
    with torch.no_grad():
        if  args.dataset == 'CAI':
            show_CAI(model.module,args)
        if  args.dataset == 'P2':
            show_P2(model.module,args)