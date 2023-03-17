from __future__ import print_function, division
import cv2
import torch
import numpy as np
import random
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler


import skimage.io
import skimage.transform
import skimage.color
import skimage



class CelebA_Dataset(Dataset):
    """ CelebA Dataset """

    def __init__(self, transform = False):
        self.img_paths = self.load_paths()    #  Load image paths
        self.annotations = self.load_annotaions()    #  Load image annotations
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = self.load_images(index)
        annot = self.annotations[index].astype(np.float32)

        sample = {'img': img, 'annot': annot}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_paths(self):
        attr_df = pd.read_csv("/content/datasets/list_attr_celeba.csv").values
        img_paths = attr_df[:2000, 0]
        root_dir = "/content/datasets/img_align_celeba/img_align_celeba/"
        return  root_dir + img_paths

    def load_images(self, index):
        img = skimage.io.imread(self.img_paths[index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0
        

    def load_annotaions(self):
        attr_df = pd.read_csv("/content/datasets/list_attr_celeba.csv").values
        print("\n - Attribute annotation:\n", attr_df[0], "\n")
        bbox_df = pd.read_csv("/content/datasets/list_bbox_celeba.csv").values
        print(" - Bounding box annotation:\n", bbox_df[0], "\n")

        attr_df[attr_df == -1] = 0
        attr = attr_df[:2000, 1:].astype(np.uint8)
        bbox = bbox_df[:2000, 1:].astype(np.uint8)

        # Transform from [x, y, w, h] to [x1, y1, x2, y2]
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        annotation = np.concatenate((bbox, attr), axis=1)
        print(" - Annotation\n", annotation[0], "\n\n")

        return annotation


    def image_aspect_ratio(self, image_index):
        image = cv2.imread(self.img_paths[image_index])
        return float(image.shape[1]) / float(image.shape[0])


class Tensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = transforms.ToTensor()(image)
        annots = torch.from_numpy(annots)
        return {'img': image, 'annot': annots}

def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 44)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:4] *= scale
        return {'img': new_image, 'annot': annots, 'scale': scale}
        # return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}



class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = np.array(annots[0]).copy()
            x2 = np.array(annots[2]).copy()
            
            x_tmp = x1.copy()

            annots[0] = cols - x2
            annots[2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key = lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
