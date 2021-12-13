from torch.utils import data
import h5py
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import torch.distributions as D
import pickle
from prdepthPytorch.UNetPytorch import *
from prdepthPytorch.utils import *
import numpy as np
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from pathlib import Path
p = Path('data/train/official')
files = sorted(p.glob('*.h5'))
print(files)

def get_midas_features(img_transform,midas):
    class SaveOutput:
        def __init__(self):
            self.outputs = []
        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out)
        def clear(self):
            self.outputs = []

    save_output = SaveOutput()
    hook_handles = []
    with torch.no_grad():
        for layer in midas.modules():
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

    with torch.no_grad():
        prediction = midas(img_transform.to(device))

    features = save_output.outputs[-13].cpu()
    del save_output

    for handle in hook_handles:
        handle.remove()
    return features,prediction



large = False
PSZ = 32
if large:
    stride = 8
    H,W = [384,512]
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
else:
    stride = 4
    H,W = [192,256]
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
    midas.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
H_patches = int((H-PSZ)/stride + 1)
W_patches = int((W-PSZ)/stride + 1)

outsize = [H_patches,W_patches]

train_images = []
train_features = []
train_depths = []
iter = 0

L1list = []
MSElist = []
loss = torch.nn.L1Loss()
mseLoss = torch.nn.MSELoss()
for h5dataset_fp in files[:200]:
    f = h5py.File(h5dataset_fp)
    image = torch.tensor(f["rgb"]).permute(1,2,0)
    depth = torch.tensor(f["depth"]).unsqueeze(0).permute(1,2,0)

    ## Downsample image and depth to size of midas inputs (192 X 256), and create overlapping patches as well
    DSimage = transform(image.numpy())
    DSdepth = transform(depth.repeat(1,1,3).numpy())[:,0,...].unsqueeze(1)
    img_patches = nn.Unfold(PSZ, stride=stride)(DSimage).view(3, PSZ, PSZ, H_patches, W_patches) #maybe add 1 to front of view for batches?
    depth_patches = nn.Unfold(PSZ, stride=stride)(DSdepth).view(1, PSZ, PSZ, H_patches, W_patches)

    ## Get feature mapping from midas
    features,prediction = get_midas_features(DSimage, midas)
    GT = center_depth(DSdepth.squeeze(0))
    predC = center_depth(-1 * prediction)
    with torch.no_grad():
        L1list.append(float(loss(GT,predC.cpu())))
        MSElist.append(np.sqrt(float(mseLoss(GT,predC.cpu()))))
    train_images.append(img_patches)
    train_features.append(features)
    train_depths.append(depth_patches)
    iter = iter +1
    gc.collect()
    print(iter)

print('Training data MIDAS L1 loss')
print(f'{np.mean(L1list)} +/ {np.std(L1list)}')
print('Training data MIDAS RMSE loss')
print(f'{np.mean(MSElist)} +/ {np.std(MSElist)}')

#Get Test data
p = Path('data/val/official')
files = sorted(p.glob('*.h5'))
val_images = []
val_features = []
val_depths = []
L1list = []
MSElist = []
for h5dataset_fp in files[:100]:
    f = h5py.File(h5dataset_fp)
    image = torch.tensor(f["rgb"]).permute(1,2,0)
    depth = torch.tensor(f["depth"]).unsqueeze(0).permute(1,2,0)

    ## Downsample image and depth to size of midas inputs (192 X 256), and create overlapping patches as well
    DSimage = transform(image.numpy())
    DSdepth = transform(depth.repeat(1,1,3).numpy())[:,0,...].unsqueeze(1)
    img_patches = nn.Unfold(PSZ, stride=stride)(DSimage).view(3, PSZ, PSZ, H_patches, W_patches) #maybe add 1 to front of view for batches?
    depth_patches = nn.Unfold(PSZ, stride=stride)(DSdepth).view(1, PSZ, PSZ, H_patches, W_patches)

    ## Get feature mapping from midas
    features, prediction = get_midas_features(DSimage, midas)
    GT = center_depth(DSdepth.squeeze(0))
    predC = center_depth(-1 * prediction)
    with torch.no_grad():
        L1list.append(float(loss(GT, predC.cpu())))
        MSElist.append(np.sqrt(float(mseLoss(GT, predC.cpu()))))
    val_images.append(img_patches)
    val_features.append(features)
    val_depths.append(depth_patches)
    gc.collect()


print('Validation data MIDAS L1 loss')
print(f'{np.mean(L1list)} +/ {np.std(L1list)}')
print('Validation data MIDAS RMSE loss')
print(f'{np.mean(MSElist)} +/ {np.std(MSElist)}')



