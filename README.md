# PIV-FlowDiffuser
This repository includes the code for the paper _PIV-FlowDiffuser: Transfer-learning-based diffusion models for particle image velocimetry_. In this work, a model incorporating diffusion denoising is utilized to make the PIV optical flow prediction more accurate. The transfer learning method is employed, which significantly reduces the model training cost.

## motivation
![figa](https://github.com/user-attachments/assets/f178ec91-4a1d-407b-93de-d60b50bdf03a)
![figb](https://github.com/user-attachments/assets/c84ef904-c024-4e85-a946-239f0de18049)

Inspired by [FlowDiffuser](https://github.com/LA30/FlowDiffuser),which includes the dual encoder, the conditional recurrent denoising decoder and the reverse denoising process, PIV-FlowDiffuser utilises a denoising diffusion probabilistic model to generate high-resolution flow fields and introduces a detailing denoising step to improve the measurement accuracy of the algorithm.The weights of the pre-trained model were transferred, and the layers associated with the scale transform were added. The model was then comprehensively fine-tuned on the PIV dataset.Since the original weights have learned the general features of optical flow estimation, it is possible to train a model dedicated to the PIV field with only a small amount of computational resources. Moreover, such a model exhibits stronger generalization and robustness.

## Install dependencies

Python 3.8 with following packages
```Shell
pytorch  2.3.1
torchvision  0.18.1
numpy  1.19.5
opencv-python  4.6.0.66
timm  0.6.12
scipy  3.6.2
matplotlib  3.3.4
```

## The experiments

&bull;![alt text](1.pdf)Show the comparison results of the operation on class 1 of the [CAI](https://github.com/shengzesnail/PIV_dataset) dataset.

&bull;![alt text](2.pdf)Show the comparison results of the operation on class 2 of the [CAI](https://github.com/shengzesnail/PIV_dataset) dataset.

&bull;![alt text](3.pdf)Show the comparison results of the operation on turbulent wavy channel flow(twcf).


