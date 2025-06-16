import math
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    dd, hh, ww = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)

    image_tensor = image.unsqueeze(0).unsqueeze(0)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image_tensor[:, :, zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]]

                with torch.no_grad():
                    pred = torch.softmax(net(test_patch), dim=1).cpu().numpy()

                score_map[:, zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]] += pred[0]
                cnt[zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]] += 1

    score_map /= cnt[None, ...]
    label_map = np.argmax(score_map, axis=0)

    return label_map

def test_single_case_fourpre(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    dd, hh, ww = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)

    image_tensor = image.unsqueeze(0).unsqueeze(0)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image_tensor[:, :, zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]]

                with torch.no_grad():
                    pred, _, _, _  = net(test_patch)
                    pred = torch.softmax(pred, dim=1).cpu().numpy()

                score_map[:, zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]] += pred[0]
                cnt[zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]] += 1

    score_map /= cnt[None, ...]
    label_map = np.argmax(score_map, axis=0)

    return label_map

def predict_one_volume(model, image, num_fore_classes, batch_size):
    # task encoding
    encoding = torch.diag(torch.ones(size = (num_fore_classes,), device="cuda", dtype=torch.int))
    
    img = image.repeat(num_fore_classes, 1, 1, 1, 1)
    # pred = [torch.ones_like(img[0, 0]) * -0.4, ]  # 0.2
    pred = [torch.zeros_like(img[0, 0]), ]
    
    for i in range(0, num_fore_classes, batch_size):
        im, en = img[i:i + batch_size], encoding[i:i + batch_size]
        output = model(im, en)
        pred += list(output.squeeze(1))
        
    return torch.stack(pred).sigmoid().cpu().numpy()

def test_single_case_Conditional(net, image, stride_xy, stride_z, patch_size, num_classes=1, batch_size=2):
    dd, hh, ww = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes,) + image.shape, dtype=np.float32)
    cnt = np.zeros(image.shape, dtype=np.float32)

    image_tensor = image.unsqueeze(0).unsqueeze(0)

    for x in range(sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image_tensor[:, :, zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]]

                with torch.no_grad():
                    pred  = predict_one_volume(net, test_patch, num_classes-1, batch_size * 2)

                score_map[:, zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]] += pred
                cnt[zs:zs+patch_size[2], ys:ys+patch_size[1], xs:xs+patch_size[0]] += 1

    score_map /= cnt[None, ...]
    label_map = np.argmax(score_map, axis=0)

    return label_map
