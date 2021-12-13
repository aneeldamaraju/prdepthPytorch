# prdepthPytorch
An implementation of the Xie et al work on patch-wise monocular depth estimation in Pytorch.

# Usage

In order to use this repository, follow the next steps
### Install dependencies

- torch
- matplotib
- h5py 
- numpy

### Run datareadNYUv2.py

 - Create a \data folder
 - Donwload [nyu_depth_v2_labeled.mat](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) and [splits.mat](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) from the NYUv2 dataset official website and put them in this data folder
 - run datareadNYUv2.py
### Run other files
Beyond this, if you want to see stats about the MIDAS model used on the splits of NYUv2, run ``loadsplits.py``.
If you want to see the implementation of the model, run ``run_UNET.ipynb``.
