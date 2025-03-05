#!/bin/bash
mkdir -p checkpoints
python -u train.py --name fd_P1 --stage CAI --validation CAI --gpus 0 --num_steps 20000 --batch_size 1 --lr 0.000005 --image_size 256 256 --wdecay 0.0002  --restore_ckpt /weights/FlowDiffuser-things.pth
python -u train.py --name fd_P2 --stage P2 --validation CAI --gpus 0 --num_steps 20000 --batch_size 1 --lr 0.000005 --image_size 256 256 --wdecay 0.0002  --restore_ckpt /weights/FlowDiffuser-things.pth