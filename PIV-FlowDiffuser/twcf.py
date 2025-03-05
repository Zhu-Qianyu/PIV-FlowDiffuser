import sys
sys.path.append('core')

import argparse
import os
import metrics
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.signal
from subprocess import call

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.tfrecord as tfrec

from piv_flowdiffuser import FlowDiffuser
from torchvision import transforms
### spline windowing
def _spline_window(window_size, power=2):
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size))) ** power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1)) ** power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


# spline windowing
cached_2d_windows = dict()


def _window_2D(window_size, power=2):
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, -1), -1)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind


class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, shard_id, num_shards, tfrecord, tfrecord_idx,
                 exec_pipelined=False, exec_async=False, is_shuffle=False, image_shape=[2, 256, 256],
                 label_shape=[2, 12]):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, exec_pipelined=False,
                                               exec_async=False)
        self.input = ops.TFRecordReader(path=tfrecord,
                                        index_path=tfrecord_idx,
                                        random_shuffle=is_shuffle,
                                        pad_last_batch=True,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        features={"target": tfrec.FixedLenFeature([], tfrec.string, ""),
                                                  "label": tfrec.FixedLenFeature([], tfrec.string, ""),
                                                  "flow": tfrec.FixedLenFeature([], tfrec.string, "")})

        self.decode = ops.PythonFunction(function=self.extract_view, num_outputs=1)
        self.reshape_image = ops.Reshape(shape=image_shape)
        self.reshape_label = ops.Reshape(shape=label_shape)

    def extract_view(self, data):
        ext_data = data.view('<f4')
        return ext_data

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = self.reshape_image(self.decode(inputs['target']))
        labels = self.reshape_label(self.decode(inputs['label']))
        flows = self.reshape_image(self.decode(inputs['flow']))

        return images, labels, flows


def data_preparation(args, rank, WORLD_SIZE, int_GPU):
    print('TWCF dataset loaded', flush=True)
    test_tfrecord = '/home/diffuser/Flowdiffuser_zqy/FlowDiffuser/data/Test_Dataset_AR_rawImage.tfrecord-00000-of-00001'
    test_tfrecord_idx = "/home/diffuser/Flowdiffuser_zqy/FlowDiffuser/data/idx_files/Test_Dataset_AR_rawImage.idx"

    # DALI data loading
    tfrecord2idx_script = "tfrecord2idx"
    if not os.path.isfile(test_tfrecord_idx):
        call([tfrecord2idx_script, test_tfrecord, test_tfrecord_idx])

    test_pipe = TFRecordPipeline(batch_size=args.batch_size_test, num_threads=8, device_id=int_GPU, num_gpus=1,
                                 tfrecord=test_tfrecord, tfrecord_idx=test_tfrecord_idx,
                                 num_shards=WORLD_SIZE, shard_id=rank,
                                 is_shuffle=False,
                                 image_shape=[2, args.image_height, args.image_width], label_shape=[12, ])
    test_pipe.build()
    test_pii = DALIGenericIterator(test_pipe, ['target', 'label', 'flow'],
                                   size=int(test_pipe.epoch_size("Reader") / WORLD_SIZE),
                                   last_batch_padded=True, fill_last_batch=False, auto_reset=True)

    return test_pii


def full_frame_folding(model, img1, img2, flow, args):
    folding_mask = torch.ones_like(img1)
    B, C, H, W = img1.size()
    NUM_Yvectors, NUM_Xvectors = int(H / args.shift - (args.offset / args.shift - 1)), \
                                 int(W / args.shift - (args.offset / args.shift - 1))
    
    predicted_flows = torch.zeros((B, 2, H, W)).cuda()

    patches_img1 = img1.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
    patches_img1 = patches_img1.reshape((-1, C, args.offset, args.offset))
    patches_img2 = img2.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
    patches_img2 = patches_img2.reshape((-1, C, args.offset, args.offset))
    flow_patches = flow.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
    flow_patches = flow_patches.reshape((-1, 2, args.offset, args.offset))

    splitted_patches_img1 = torch.split(patches_img1, args.split_size, dim=0)
    splitted_patches_img2 = torch.split(patches_img2, args.split_size, dim=0)
    splitted_flow_patches = torch.split(flow_patches, args.split_size, dim=0)

    WINDOW_SPLINE_2D = torch.from_numpy(np.squeeze(_window_2D(window_size=args.offset, power=2))).cuda()

    with torch.no_grad():
        predicted_flow_patches = predicted_flows.unfold(3, args.offset, args.shift) \
            .unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
        predicted_flow_patches = predicted_flow_patches.reshape((-1, 2, args.offset, args.offset))
        splitted_predicted_flow_patches = torch.split(predicted_flow_patches, args.split_size, dim=0)
        splitted_flow_output_patches = []

        for split in range(len(splitted_patches_img1)):
            # print(f"splitted_patches_img1[split] shape: {splitted_patches_img1[split].shape}")
            flow_low, flow_prediction = model(splitted_patches_img1[split], splitted_patches_img2[split],
                                              flow_init=splitted_predicted_flow_patches[split], iters=args.iters,
                                              test_mode=True)
            all_flow_iters = flow_prediction[-1]
            splitted_flow_output_patches.append(all_flow_iters)

        flow_output_patches = torch.cat(splitted_flow_output_patches, dim=0)
        flow_output_patches = flow_output_patches * WINDOW_SPLINE_2D
        flow_output_patches = flow_output_patches.reshape(
            (B, NUM_Yvectors, NUM_Xvectors, 2, args.offset, args.offset)).permute(0, 3, 1, 2, 4, 5)
        flow_output_patches = flow_output_patches.contiguous().view(B, 2, -1, args.offset * args.offset)
        flow_output_patches = flow_output_patches.permute(0, 1, 3, 2)
        flow_output_patches = flow_output_patches.contiguous().view(B, 2 * args.offset * args.offset, -1)
        predicted_flows_iter = F.fold(flow_output_patches, output_size=(H, W), kernel_size=args.offset,
                                      stride=args.shift)

        folding_mask = folding_mask[:, :2, :, :]
        mask_patches = folding_mask.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift)
        mask_patches = mask_patches.contiguous().view(B, 2, -1, args.offset, args.offset)
        mask_patches = mask_patches * WINDOW_SPLINE_2D
        mask_patches = mask_patches.view(B, 2, -1, args.offset * args.offset)
        mask_patches = mask_patches.permute(0, 1, 3, 2)
        mask_patches = mask_patches.contiguous().view(B, 2 * args.offset * args.offset, -1)
        folding_mask = F.fold(mask_patches, output_size=(H, W), kernel_size=args.offset, stride=args.shift)

        predicted_flows += predicted_flows_iter / folding_mask

    return predicted_flows

def show_dataset(model, args, rank, WORLD_SIZE, int_GPU):
    model.eval()

    test_pii = data_preparation(args, rank, WORLD_SIZE, int_GPU)

    result_folder_gt = os.path.join('result_1', 'flow_gt')
    result_folder_pred = os.path.join('result_1', 'flow')
    os.makedirs(result_folder_gt, exist_ok=True)
    os.makedirs(result_folder_pred, exist_ok=True)
    PIV_results_TWCF = np.load('/home/diffuser/Flowdiffuser_zqy/FlowDiffuser/data/PIV_results_TWCF.npy')
    mask_TWCF = np.load('/home/diffuser/Flowdiffuser_zqy/FlowDiffuser/data/mask_TWCF.npy')
    print(f"Number of samples in dataset: {test_pii._size}")
    epe = []

    for i_batch, sample_batched in enumerate(test_pii):
        if i_batch >= 30:
            break
        local_dict = sample_batched[0]
        images = local_dict['target'].type(torch.FloatTensor).cuda() / 256
        flows = local_dict['flow'].type(torch.FloatTensor).cuda()

        img1 = images[:, 0:1, :, :]
        img2 = images[:, 1:2, :, :]
        # print(img1.shape)
        
        img1 = img1.repeat(1, 3, 1, 1)
        img2 = img2.repeat(1, 3, 1, 1)
        # print(img1.shape)

        flow_prediction = full_frame_folding(model, img1, img2, flows, args)

        U_PascalPIV = PIV_results_TWCF[i_batch, 0, :, :]
        V_PascalPIV = PIV_results_TWCF[i_batch, 1, :, :]
        
        flow = flows.squeeze(0)
        flow_prediction = flow_prediction.squeeze(0)
        x, y = np.meshgrid(np.arange(flow.shape[2]), np.arange(flow.shape[1]), indexing="ij")

        u_gt = U_PascalPIV
        v_gt = V_PascalPIV
        if args.modty == 1:
            u_pd = flow_prediction[1].detach().cpu()
            v_pd = flow_prediction[0].detach().cpu()
        else :
            u_pd = flow_prediction[0].detach().cpu()
            v_pd = flow_prediction[1].detach().cpu()

        plt.rcParams['text.usetex'] = False
        plt.rcParams['pdf.fonttype'] = 42  

        plt.figure(num=None, figsize=(9, 6),  facecolor='w', edgecolor='k')
        plt.subplot(2, 2, 1)
        plt.pcolor(np.squeeze(U_PascalPIV), cmap='viridis', vmin=-2, vmax=12,rasterized = True)
        plt.axis('off')
        cbar = plt.colorbar()
        # cbar.ax.set_ylabel('displacement [px]', fontsize=14)
        # t = plt.text(0, 505, 'PascalPIV', fontsize=16)
        # t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        plt.subplot(2, 2, 2)
        plt.pcolor(np.squeeze(V_PascalPIV), cmap='viridis', vmin=-1, vmax=1,rasterized = True)
        plt.axis('off')
        cbar = plt.colorbar()
        # cbar.ax.set_ylabel('displacement [px]', fontsize=14)
        plt.subplot(2, 2, 3)
        plt.pcolor(u_pd * mask_TWCF, cmap='viridis', vmin=-2, vmax=12,rasterized = True)
        plt.axis('off')
        cbar = plt.colorbar()
        # cbar.ax.set_ylabel('displacement [px]', fontsize=14)
        # t = plt.text(0, 2025, 'PIV_fds', fontsize=16)
        # t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        plt.subplot(2, 2, 4)
        plt.pcolor(v_pd * mask_TWCF, cmap='viridis', vmin=-1, vmax=1,rasterized = True)
        plt.axis('off')
        cbar = plt.colorbar()
        # cbar.ax.set_ylabel('displacement [px]', fontsize=14)
        plt.savefig(f"result/u_v_gt_pred_twcf{i_batch:03d}.png")
        plt.savefig(f"result/u_v_gt_pred_twcf{i_batch:03d}.pdf")
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--modty', help="restore checkpoint type")
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--offset', default=256, type=int, help='interrogation window size')
    parser.add_argument('--shift', default=64, type=int, help='shift of interrogation window in px')
    parser.add_argument('--split_size', default=1, type=int)
    parser.add_argument('--iters', default=12, type=int, help='number of update steps in ConvGRU')
    parser.add_argument('--mixed_precision', action='store_true', default=False, help='use mixed precision')
    args = parser.parse_args()

    args.test_dataset = 'twcf'
    args.image_height = 2160
    args.image_width = 2560

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()

    rank = 0
    WORLD_SIZE = 1
    int_GPU = 0
    plt.rcParams['text.usetex'] = False
    plt.rcParams['pdf.fonttype'] = 42  
    with torch.no_grad():
        show_dataset(model, args, rank, WORLD_SIZE, int_GPU)