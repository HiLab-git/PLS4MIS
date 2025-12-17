'''
Define the dataset transforms for Abdomen Organ segmentation task
'''
import os
import torch
import numpy as np
from glob import glob
import random
import h5py
import itertools
from torch.utils.data.sampler import Sampler


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pw = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, h, w) = image.shape

        d1 = int(round((d - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        w1 = int(round((w - self.output_size[2]) / 2.))

        label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]

        return {'image': image,
                'label': label,
                'img_name': sample['img_name']}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pw = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, h, w) = image.shape

        d1 = np.random.randint(0, d - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        w1 = np.random.randint(0, w - self.output_size[2])

        label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]

        return {'image': image,
                'label': label,
                'img_name': sample['img_name']}


class RandomFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        axis = np.random.randint(1, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image,
                'label': label,
                'img_name': sample['img_name']}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise

        return {'image': image,
                'label': label,
                'img_name': sample['img_name']}


class LabeledClass(object):
    """
    For partially labeled segmentation:
    Generate a fixed-length one-hot vector indicating which classes are annotated.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes - 1  # no background(0)

    def __call__(self, sample):
        label = sample['label']
        target_class = np.unique(label)
        target_class = target_class[target_class != 0]  # remove background

        # generate one-hot vector
        one_hot = np.zeros(self.num_classes, dtype=np.float32)
        one_hot[target_class - 1] = 1.0   # Note: Category IDs generally start from 1, so (-1)

        return {
            'image': sample['image'],
            'label': label,
            'img_name': sample['img_name'],
            'cur_task': one_hot
        }


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)

        return {'image': image,
                'label': sample['label'],
                'onehot_label': onehot_label,
                'img_name': sample['img_name'],
                'cur_task': sample['cur_task']}


class TrainerCrop(object):
    """
    Crop randomly the image in Abdomen Organ sample
    Args:
    output_deep (int): Desired output deep
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        while True:
            slices = []
            for dim, patch in zip(image.shape, self.patch_size[::-1]):
                start = random.randint(0, dim - patch)
                end = start + patch
                slices.append(slice(start, end))

            cropped_image = image[slices[0], slices[1], slices[2]]
            cropped_label = label[slices[0], slices[1], slices[2]]
            
            if np.max(cropped_image) != np.min(cropped_image):
                break

        return {'image': cropped_image,
                'label': cropped_label,
                'img_name': sample['img_name'],
                'cur_task': sample['cur_task']}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image = sample['image']
        label = sample['label']

        # max and min 0-1
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # add channel dim
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        image, label = image.astype(np.float32), label.astype(np.float32)

        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'onehot_label': torch.from_numpy(sample['onehot_label']),
                'cur_task': torch.from_numpy(sample['cur_task']),
                'img_name': sample['img_name'],}


class Test_ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        image = sample['image']
        label = sample['label']

        # max and min 0-1
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # add channel dim
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

        image, label = image.astype(np.float32), label.astype(np.float32)

        return {'img_name': sample['img_name'],
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
