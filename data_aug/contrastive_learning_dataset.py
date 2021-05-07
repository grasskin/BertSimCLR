from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torch.utils.data import Dataset
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer

import random
import os
import json

class CocoDetectionBaseline(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = [target[0]]

        return img, target


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, tokenizer, transform=None, target_transform=None, generateEncodings=True, encodingPath=''):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.tokenizer = tokenizer



        if generateEncodings:
            ann_ids = self.coco.getAnnIds(imgIds=self.ids)
            targets = self.coco.loadAnns(ann_ids)
            encodings = {}
            jsonPath = '/'.join(annFile.split('/')[:-2]) + '/encoded_' + annFile.split('/')[-1]
            
            print('Starting to generate sentence encoding')
            for t in tqdm(targets):
                encodings[t['caption']] = self.tokenizer.encode(t['caption']).tolist()
            print('Generated sentence encoding')
            with open(jsonPath, 'w') as outfile:
                json.dump(encodings, outfile)
            print('Saved to', jsonPath)
        else:
            self.targets = json.load(open(encodingPath, 'r'))
            for target in self.targets:
                self.targets[target] = np.array(self.targets[target], dtype=np.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = random.choice(target)

        target_encoding = torch.from_numpy(self.targets[target['caption']])

        return (*img, target_encoding)


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def coco_collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image_aug_1: torch tensor of shape (3, 96, 96).
            - image_aug_2: torch tensor of shape (3, 96, 96).
            - caption: (768,)
    Returns:
        images: torch tensor of shape (batch_size, 3, 96, 96).
        images2: torch tensor of shape (batch_size, 3, 96, 96).
        caption: torch tensor of shape (batch_size, 768)
    """
    # Sort a data list by caption length (descending order).
    images, images2, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    images2 = torch.stack(images2, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    captions = torch.stack(captions, 0)

    return images, images2, captions


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'mscoco': lambda: CocoDetection(os.path.join(self.root_folder, 'mscoco', 'train2017'), 
                                            annFile = os.path.join(self.root_folder, 'mscoco', 'annotations', 'captions_train2017.json'),
                                            tokenizer= SentenceTransformer('bert-base-nli-mean-tokens'),
                                            transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                              encodingPath=os.path.join(self.root_folder,'mscoco', 'encoded_captions_train2017.json'),
                                                              generateEncodings=False),
                          'mscocovalid': lambda: CocoDetection(os.path.join(self.root_folder, 'mscoco', 'val2017'), 
                                            annFile = os.path.join(self.root_folder, 'mscoco', 'annotations', 'captions_val2017.json'),
                                            tokenizer= SentenceTransformer('bert-base-nli-mean-tokens'),
                                            transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                              encodingPath=os.path.join(self.root_folder,'mscoco', 'encoded_captions_val2017.json'),
                                                              generateEncodings=False),
                          'mscocobaseline': lambda: CocoDetectionBaseline(os.path.join(self.root_folder, 'mscoco', 'train2017'), 
                                            annFile = os.path.join(self.root_folder, 'mscoco', 'annotations', 'captions_train2017.json'),
                                            transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views)),
                          'mscocobaselinevalid': lambda: CocoDetectionBaseline(os.path.join(self.root_folder, 'mscoco', 'val2017'), 
                                            annFile = os.path.join(self.root_folder, 'mscoco', 'annotations', 'captions_val2017.json'),
                                            transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views))}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
