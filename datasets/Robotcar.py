import setlog
import torch.utils as utils
import torch.utils.data
import torchvision as torchvis
import datasets.multmodtf as tf
import pandas as pd
import tqdm
import PIL.Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os


logger = setlog.get_logger(__name__)


class VBLDataset(utils.data.Dataset):
    def __init__(self, root, coord_file, modalities, **kwargs):
        self.root = root
        self.transform = kwargs.pop('transform', 'default')
        self.bearing = kwargs.pop('bearing', True)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = {
                'first': (tf.Resize((224, 224)), tf.ToTensor())
            }

        self.coord = pd.read_csv(self.root + coord_file, header=None, sep=',', dtype=np.float64) if self.bearing \
            else pd.read_csv(self.root + coord_file, header=None, sep='\t', dtype=np.float64)

        self.modalities = dict()
        for mod_name in modalities:
            self.modalities[mod_name] = pd.read_csv(self.root + modalities[mod_name], header=None)

        self.used_mod = self.modalities.keys()

    def __len__(self):
        return self.coord.__len__()

    def __getitem__(self, idx):
        sample = dict()
        for mod_name in self.used_mod:
            file_name = self.root + self.modalities[mod_name].ix[idx, 0]
            sample[mod_name] = PIL.Image.open(file_name)

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod]})[mod]

        sample['coord'] = self.coord.ix[idx, 0:2].as_matrix().astype('float') if self.bearing \
            else self.coord.ix[idx, 0:1].as_matrix().astype('float')

        return sample


class TripletDataset(utils.data.Dataset):
    def __init__(self, **kwargs):

        self.main = kwargs.pop('main', None)
        self.examples = kwargs.pop('examples', None)
        self.num_positive = kwargs.pop('num_positives', 4)
        self.num_negative = kwargs.pop('num_negatives', 20)
        self.num_triplets = kwargs.pop('num_triplets', 1000)
        self.max_pose_dist = kwargs.pop('max_pose_dist', 7)        # meters
        self.min_neg_dist = kwargs.pop('min_neg_dist', 700)         # meters
        self.max_angle = kwargs.pop('max_angle', 0.174533)          # radians, 20 degrees
        load_triplets = kwargs.pop('load_triplets', None)
        self._used_mod = kwargs.pop('used_mod', ['rgb'])

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if load_triplets:
            self.triplets = torch.load(os.environ['DATASET'] + load_triplets)
        else:
            logger.info('Creating {} triplets...'.format(self.num_triplets))
            self.triplets = self.build_triplets()

    def __len__(self):
        return self.triplets.__len__()

    def build_triplets(self):
        triplets = list()
        for i in np.random.choice(range(len(self.main.coord)), size=len(self.main.coord), replace=False):
            q = self.main.coord.values[i]
            triplet = {
                'positives': [],
                'negatives': []
            }
            for num_ex, example in enumerate(self.examples):
                for idx, ex in enumerate(example.coord.values):
                    dist = np.linalg.norm([q[0] - ex[0], q[1] - ex[1]])
                    if dist > self.min_neg_dist:
                        triplet['negatives'].append([num_ex, idx])
                    elif dist < self.max_pose_dist:
                        ang = abs(q[2] - ex[2])
                        if ang < self.max_angle:
                            triplet['positives'].append([num_ex, idx])

            if triplet['positives'] and triplet['negatives'] \
                    and len(triplet['positives']) >= self.num_positive \
                    and len(triplet['negatives']) >= self.num_negative:
                triplet['query'] = i
                np.random.shuffle(triplet['negatives'])  # Random shuffling to have diversity when calling
                np.random.shuffle(triplet['positives'])  # Random shuffling to have diversity when calling
                triplets.append(triplet)
                logger.debug('New triplet with {} positives and {} negatives'.format(len(triplet['positives']),
                                                                                     len(triplet['negatives'])))
                logger.debug('Totoal number of triplets {}'.format(len(triplets)))
                if len(triplets) == self.num_triplets:
                    break

        return triplets

    def __getitem__(self, idx):
        sample = {
            'query': self.main[self.triplets[idx]['query']],
            'positives': [self.examples[self.triplets[idx]['positives'][i][0]][self.triplets[idx]['positives'][i][1]]
                          for i in range(self.num_positive)],
            'negatives': [self.examples[self.triplets[idx]['negatives'][i][0]][self.triplets[idx]['negatives'][i][1]]
                          for i in range(self.num_negative)]
        }
        return sample

    @property
    def used_mod(self):
        return self._used_mod

    @used_mod.setter
    def used_mod(self, mods):
        self.main.used_mod = mods
        for data in self.examples:
            data.used_mod = mods
        self._used_mod = mods


def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    buffer = tuple()
    for name, mod in sample_batched.items():
        if name not in ('coord',):
            min_v = mod.min()
            mod -= min_v
            max_v = mod.max()
            mod /= max_v
            buffer += (mod,)

    images_batch = torch.cat(buffer, 0)
    grid = torchvis.utils.make_grid(images_batch, nrow=4)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    root_to_folders = os.environ['ROBOTCAR'] + 'training/TrainDataset_02_10_15/'

    modtouse = {'rgb': 'dataset.txt', 'depth': 'depth_dataset.txt', 'ref': 'ref_dataset.txt'}
    transform = {
        'first': (tf.Resize(280), tf.RandomCrop(224)),
        'rgb': (tf.ToTensor(), ),
        'depth': (tf.ToTensor(),),
        'ref': (tf.ToTensor(),)
    }

    """
    dataset = VBLDataset(root=root_to_folders,
                         modalities=modtouse,
                         coord_file='coordxImbearing.txt',
                         transform=transform)

    dataloader = utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        print(b['coord'])
        plt.show()
    """

    dataset_1 = VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_05_19_15/',
                           modalities=modtouse,
                           coord_file='coordxImbearing.txt',
                           transform=transform)
    dataset_2 = VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_08_28_15/',
                           modalities=modtouse,
                           coord_file='coordxImbearing.txt',
                           transform=transform)
    dataset_3 = VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_11_10_15/',
                           modalities=modtouse,
                           coord_file='coordxImbearing.txt',
                           transform=transform)

    triplet_dataset = TripletDataset(main=dataset_1, examples=[dataset_2, dataset_3],
                                     num_triplets=400, num_positives=4, num_negatives=20)
    torch.save(triplet_dataset.triplets, '400triplets.pth')
    dtload = utils.data.DataLoader(triplet_dataset, batch_size=4)

    for b in dtload:
        plt.figure(1)
        show_batch(b['query'])
        plt.figure(2)
        show_batch(b['positives'][0])
        plt.figure(3)
        show_batch(b['negatives'][0])
        plt.show()
