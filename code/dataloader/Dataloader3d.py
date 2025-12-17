'''
Define a dataset loader for Abdomen Organ segmentation task
'''
import sys
import os, logging, torch
import numpy as np
import random
import SimpleITK as sitk
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms

class AbdomenOrgan(Dataset):
    def __init__(self, nii_dir='./datasets/WORD', mode='test', transform=None):
        '''
        :param nii_dir:  Data storage location.
        :param mode:  Mode in train, valid and test.
        '''
        self.nii_dir = nii_dir
        self.mode = mode
        self.transform = transform
        self.sample_list = []

        if self.mode == 'train':
            image_dir = os.path.join(self.nii_dir, 'imagesTr/')
            print('==> Loading {} data from: {}'.format(mode, image_dir))

            image_list = glob(image_dir + '*.nii.gz')
            for image_path in image_list:
                gt_path = image_path.replace('imagesTr', 'labelsTr')
                self.sample_list.append({'image': image_path, 'label': gt_path})
        elif self.mode == 'test':
            image_dir = os.path.join(self.nii_dir, 'imagesTs/')
            print('==> Loading {} data from: {}'.format(mode, image_dir))

            image_list = glob(image_dir + '*.nii.gz')
            for image_path in image_list:
                gt_path = image_path.replace('imagesTs', 'labelsTs')
                self.sample_list.append({'image': image_path, 'label': gt_path})
        elif self.mode == 'val':
            image_dir = os.path.join(self.nii_dir, 'imagesVal/')
            print('==> Loading {} data from: {}'.format(mode, image_dir))

            image_list = glob(image_dir + '*.nii.gz')
            for image_path in image_list:
                gt_path = image_path.replace('imagesVal', 'labelsVal')
                self.sample_list.append({'image': image_path, 'label': gt_path})
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        image_name = self.sample_list[index]
        _dataimg = sitk.ReadImage(image_name['image'])
        _image = sitk.GetArrayFromImage(_dataimg)
        _datagt = sitk.ReadImage(image_name['label'])
        _target = sitk.GetArrayFromImage(_datagt)
        _img_name = image_name['image'].split('/')[-1]

        sample = {'image': _image, 'label': _target, 'img_name': _img_name}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample
