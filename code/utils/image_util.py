import random
import numbers
import torch
import numpy as np
import torch.nn as nn
from scipy import ndimage

def itensity_normalize(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized n                                                                                                                                                                 d volume
    """

    # pixels = volume[volume > 0]
    mean = volume.mean()
    std = volume.std()
    out = (volume - mean) / std
    # out_random = np.random.normal(0, 1, size=volume.shape)
    # out[volume == 0] = out_random[volume == 0]

    return out

def get_largest_k_components(image, k = 1):
    """
    Get the largest K components from 2D or 3D binary image.

    :param image: The input ND array for binary segmentation.
    :param k: (int) The value of k.

    :return: An output array (k == 1) or a list of ND array (k>1) 
        with only the largest K components of the input. 
    """
    dim = len(image.shape)
    if(image.sum() == 0 ):
        print('the largest component is null')
        return image
    s = ndimage.generate_binary_structure(dim,1)      
    labeled_array, numpatches = ndimage.label(image, s)
    sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
    sizes_sort = sorted(sizes, reverse = True)
    kmin = min(k, numpatches)
    output = []
    for i in range(kmin):
        labeli = np.where(sizes == sizes_sort[i])[0] + 1
        output_i = np.asarray(labeled_array == labeli, np.uint8)
        output.append(output_i)
    return  output[0] if k == 1 else output

def get_multi_class_components(pseudo_label):
    bs, d, h, w = pseudo_label.shape
    largest_regions = np.zeros_like(pseudo_label)
    
    for b in range(bs):
        for i in np.unique(pseudo_label[b]):
            if i == 0:
                continue

            mask = pseudo_label[b] == i
            if i not in [10, 11, 12]:
                mask_largest_region = get_largest_k_components(mask, k=1) * i
            else:
                mask_largest_region = mask * i
            
            largest_regions[b] += mask_largest_region
    
    return torch.from_numpy(largest_regions)

def Label_fusion(old_label, pseudo_label):
    new_label = old_label.detach()
    bs, d, h, w = old_label.shape
    
    for b in range(bs): 
        pseudo_label_element = torch.unique(pseudo_label[b])
        old_label_element = torch.unique(old_label[b])
        if 0 in pseudo_label_element:
            pseudo_label_element = pseudo_label_element[pseudo_label_element != 0]
        for j in pseudo_label_element:
            if j in old_label_element:
                pass
            else:
                useful_data = torch.where(pseudo_label[b] == j, j, 0)
                change_data = torch.where(new_label[b] == 0, 1, 0)
                useful_data = useful_data * change_data
                new_label[b] = new_label[b] + useful_data
    
    return new_label
