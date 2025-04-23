#!/bin/bash
python -u show.py --model /weights/PIV-flowdiffusers.pth --dataset CAI --savedir result_show0 --modty 1
python -u show.py --model /weights/PIV-flowdiffusers.pth --dataset P2 --savedir result_show1 --modty 1
python -u show.py --model /weights/PIV-flowdiffuser-class1.pth --dataset CAI --savedir result_show2 --modty 1
python -u show.py --model /weights/PIV-flowdiffuser-class1.pth --dataset P2 --savedir result_show3 --modty 1
python -u show.py --model /weights/PIV-flowdiffuser-class2.pth --dataset CAI --savedir result_show4 --modty 2
python -u show.py --model /weights/PIV-flowdiffuser-class2.pth --dataset P2 --savedir result_show5 --modty 2

python -u evaluate_detail.py --model /weights/PIV-flowdiffusers.pth --dataset CAI --modty 1
python -u evaluate_detail.py --model /weights/PIV-flowdiffuser-class1.pth --dataset CAI --modty 1
python -u evaluate_detail.py --model /weights/PIV-flowdiffuser-class2.pth --dataset CAI --modty 2

python -u eval_all.py --model /weights/PIV-flowdiffusers.pth --dataset CAI --modty 1
python -u eval_all.py --model /weights/PIV-flowdiffusers.pth --dataset P2 --modty 1
python -u eval_all.py --model /weights/PIV-flowdiffuser-class1.pth --dataset CAI --modty 1
python -u eval_all.py --model /weights/PIV-flowdiffuser-class1.pth --dataset P2 --modty 1
python -u eval_all.py --model /weights/PIV-flowdiffuser-class2.pth --dataset CAI --modty 2
python -u eval_all.py --model /weights/PIV-flowdiffuser-class2.pth --dataset P2 --modty 2

python -u twcf.py --model 
