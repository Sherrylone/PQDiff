from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import pandas as pd
import os
import glob
import einops
import torchvision.transforms.functional as F
from dataset.pos import get_2d_local_sincos_pos_embed

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError

class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size)
        self.resolution = resolution
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp"]:
            results.append(full_path)
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class RandomResizedCropCoord(transforms.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=Image.BICUBIC):
        self.size = size
        self.ratio = ratio
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, (self.size, self.size), self.interpolation)
        return (i, j, h, w), img

    def __call__(self, img):
        return self.forward(img)

class WikiArt(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        f_name = os.listdir(path)
        train_files = [path+str(f_name[i]) for i in range(len(f_name))]
        class_names = [0 for i in range(len(train_files))]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting WikiArt files, total images: %d' % len(train_files))

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size, anchor_crop_scale=(0.2, 0.5), target_crop_scale=(0.8, 1.0))

        self.resolution = resolution

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class Flickr(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        print(f'Counting FLickr files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        f_name = os.listdir(path)
        train_files = [path+str(f_name[i]) for i in range(len(f_name)) if int(f_name[i].split('_')[-1].split('.')[0].replace(',', ''))<=5040]
        # train_files = [path+str(f_name[i]) for i in range(len(f_name))]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting FLickr files, total images: %d' % len(train_files))

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size, anchor_crop_scale=(0.2, 0.5), target_crop_scale=(0.8, 1.0))

        # val_files = _list_image_files_recursively(path)
        # train_labels = [sorted_classes[x] for x in class_names]
        self.resolution = resolution

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class Building(DatasetFactory):
    def __init__(self, path, resolution, embed_dim, grid_size):
        super().__init__()

        print(f'Counting Building files from {path}')
        train_files = _list_image_files_recursively(path)
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        f_name = os.listdir(path)
        # train_files = [path+str(f_name[i]) for i in range(len(f_name)) if int(f_name[i].split('_')[-1].split('.')[0].replace(',', ''))<=5040]
        train_files = [path+str(f_name[i]) for i in range(len(f_name))]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting Building files, total images: %d' % len(train_files))

        self.train = ImageDataset(resolution, train_files, train_labels, embed_dim, grid_size, anchor_crop_scale=(0.2, 0.5), target_crop_scale=(0.8, 1.0))

        # val_files = _list_image_files_recursively(path)
        # train_labels = [sorted_classes[x] for x in class_names]
        self.resolution = resolution

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        embed_dim,
        grid_size,
        anchor_crop_scale=(0.15, 0.40),
        target_crop_scale=(0.8, 1.0)
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = []
        for i in range(len(image_paths)):
            try:
                pil_image = Image.open(image_paths[i])
                pil_image.load()
                self.image_paths.append(image_paths[i])
            except:
                pass
        self.labels = labels
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.anchor_rcr = RandomResizedCropCoord(resolution, scale=anchor_crop_scale, interpolation=transforms.InterpolationMode.BILINEAR)
        self.target_rcr = RandomResizedCropCoord(resolution, scale=target_crop_scale, interpolation=transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return len(self.image_paths)

    def calculate_sin_cos(self, lpos, gpos):
        kg = gpos[3] / self.grid_size
        w_bias = (lpos[1] - gpos[1]) / kg
        kl = lpos[3] / self.grid_size
        w_scale = kl / kg
        kg = gpos[2] / self.grid_size
        h_bias = (lpos[0] - gpos[0]) / kg
        kl = lpos[2] / self.grid_size
        h_scale = kl / kg
        return get_2d_local_sincos_pos_embed(self.embed_dim, self.grid_size, w_bias, w_scale, h_bias, h_scale)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        anchor_pos, anchor_img = self.anchor_rcr(pil_image)
        target_pos, target_img = self.target_rcr(pil_image)
        target_pos = self.calculate_sin_cos(target_pos, anchor_pos)
        anchor_img = np.array(anchor_img) / 127.5 - 1
        target_img = np.array(target_img) / 127.5 - 1
        return np.transpose(target_img, [2,0,1]), np.transpose(anchor_img, [2,0,1]), target_pos


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


def get_dataset(name, **kwargs):
    if name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'flickr':
        return Flickr(**kwargs)
    elif name == 'wikiart':
        return WikiArt(**kwargs)
    elif name == 'building':
        return Building(**kwargs)
    else:
        raise NotImplementedError(name)
