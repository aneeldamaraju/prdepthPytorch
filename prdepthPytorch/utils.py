import numpy as np
import torch

def reconstruct_img(out,H,W,stride,PSZ):
    rec = np.zeros([H,W,int(out.shape[0]*out.shape[1])])
    num_overlaps = np.zeros([H,W])
    itr = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            rec[i*stride:i*stride + PSZ,j*stride:j*stride + PSZ,itr] = out[i,j,...].detach().cpu().numpy()
            num_overlaps[i*stride:i*stride + PSZ,j*stride:j*stride + PSZ] += 1
            itr = itr+1
    return np.divide(np.sum(rec,axis = -1),num_overlaps)

def center_depth(gt_depth):
    max_d = torch.max(gt_depth)
    min_d = torch.min(gt_depth)
    range_d = max_d-min_d
    gt2 = (gt_depth-(max_d-range_d/2))/(range_d/2)
    return gt2

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
    return features