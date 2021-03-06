import setlog
import torch.utils as utils
import torch.utils.data
import torchvision as torchvis
import datasets.multmodtf as tf
import pandas as pd
import PIL.Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import time


logger = setlog.get_logger(__name__)


class VBLDataset(utils.data.Dataset):
    def __init__(self, root, coord_file, modalities, **kwargs):
        self.root = root
        self.transform = kwargs.pop('transform', 'default')
        self.bearing = kwargs.pop('bearing', True)
        self.K_fisheye = kwargs.pop('K_fisheye', [[340, 0, 435], [0.0, 340, 435], [0.0, 0.0, 1.0]])
        self.K_stereo = kwargs.pop('K_stereo', [[964.82, 0, 643.78], [0.0, 964.82, 484.40], [0.0, 0.0, 1.0]])
        self.K_CMU = kwargs.pop('K_stereo', [[406.298, 0, 312.809], [0.0, 407.497, 233.289], [0.0, 0.0, 1.0]])

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
            if 'img' in file_name:
                sample['K'] = np.array(self.K_CMU, dtype=np.float32)
            elif 'mono' in file_name:
                sample['K'] = np.array(self.K_fisheye, dtype=np.float32)
            elif 'centre' in file_name:
                sample['K'] = np.array(self.K_stereo, dtype=np.float32)
            else:
                raise AttributeError('Unknown camera calibration for image {}'.format(file_name))

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod],
                                                                                    'K': sample['K']})[mod]

        sample['coord'] = self.coord.ix[idx, 0:2].values if self.bearing \
            else self.coord.ix[idx, 0:1].values

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
        self.ex_shuffle = kwargs.pop('ex_shuffle', True)
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
                logger.debug('Total number of triplets {}'.format(len(triplets)))
                if len(triplets) == self.num_triplets:
                    break

        return triplets

    def __getitem__(self, idx):
        if self.ex_shuffle:
            np.random.shuffle(self.triplets[idx]['positives'])
            np.random.shuffle(self.triplets[idx]['negatives'])
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
        if name not in ('coord', 'K'):
            '''
            min_v = mod.min()
            mod -= min_v
            max_v = mod.max()
            mod /= max_v
            '''
            buffer += (mod,)

    images_batch = torch.cat(buffer, 0)
    grid = torchvis.utils.make_grid(images_batch, nrow=4)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    root_to_folders = os.environ['ROBOTCAR'] + 'training/TrainDataset_02_10_15/'

    modtouse = {'rgb': 'dataset.txt'}#, 'mono_depth': 'mono_depth_dataset.txt', 'depth': 'depth_dataset.txt'}#, 'ref': 'mono_ref_dataset.txt'}
    transform = {
        'first': (tf.Resize((420, 420)),),
        'rgb': (tf.ToTensor(), ),
        #'depth': (tf.ToTensor(), tf.DepthTransform(depth_factor=1e-4, replacing_value=100000, error_value=0, inverse=True), tf.JetTransform(s_lut=1024)),
        'mono_depth': (tf.ToTensor(), tf.JetTransform(s_lut=1024)),
        'ref': (tf.ToTensor(), tf.DepthTransform(depth_factor=1e-3), tf.JetTransform(s_lut=1024))
    }
    transform_wo_q = {
        'first': (tf.Resize((224, 224)),),
        'rgb': (tf.ToTensor(),),
        #'mono_depth': (tf.ToTensor(), tf.DepthTransform(depth_factor=1.0, error_value=0.0, replacing_value=1.0), tf.Normalize(mean=[0.2291], std=[1]), tf.JetTransform()),
        #'depth': (tf.ToTensor(),),
        #'ref': (tf.ToTensor(), tf.Normalize(mean=[0.2164], std=[0.08]))
    }


    #os.environ['CMU'] = "/mnt/anakim/data/"

    dataset_1 = VBLDataset(root=os.environ['CMU'] + 'sunny/',
                           modalities={'rgb': 'dataset.txt'},
                           coord_file='coordxImbearing.txt',
                           transform=transform_wo_q)

    dataset_2 = VBLDataset(root=os.environ['CMU'] + 'snow/',
                           modalities={'rgb': 'dataset.txt'},
                           coord_file='coordxImbearing.txt',
                           transform=transform_wo_q)

    dataset_3 = VBLDataset(root=os.environ['CMU'] + 'autumn/',
                           modalities={'rgb': 'dataset.txt'},
                           coord_file='coordxImbearing.txt',
                           transform=transform_wo_q)

    dataset_4 = VBLDataset(root=os.environ['CMU'] + 'long_term/',
                           modalities={'rgb': 'dataset.txt'},
                           coord_file='coordxImbearing.txt',
                           transform=transform_wo_q)
    '''

    dataset_1 = VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_05_19_15/',
                           modalities={'rgb': 'pruned_dataset.txt', 'depth': 'pruned_true_depth_dataset.txt',
                                        'ref': 'pruned_true_ref_dataset.txt'},
                           coord_file='pruned_coordxImbearing.txt',
                           transform=transform)

    dataset_2 = VBLDataset(root=os.environ['ROBOTCAR'] + 'training/TrainDataset_05_19_15/',
                           modalities={'rgb': 'dataset.txt', 'depth': 'true_depth_dataset.txt',
                                       'ref': 'true_ref_dataset.txt'},
                           coord_file='coordxImbearing.txt',
                           transform=transform)
    '''
    triplet_dataset = TripletDataset(main=dataset_1, examples=[dataset_2, dataset_3], max_pose_dist=2,
                                     num_triplets=20, num_positives=4, num_negatives=4, ex_shuffle=True)
    #%torch.save(triplet_dataset.triplets, 'night_200_triplets.pth')
    dtload = utils.data.DataLoader(triplet_dataset, batch_size=4)


    for b in dtload:
        plt.figure(1)
        show_batch(b['query'])
        plt.figure(2)
        show_batch(b['positives'][0])
        plt.figure(3)
        show_batch(b['negatives'][1])
        '''
        plt.figure(0)
        images_batch = torch.cat((b['query']['rgb'],
                                  b['positives'][0]['rgb'],
                                  b['positives'][1]['rgb'],
                                  b['positives'][2]['rgb'],), 0)
        grid = torchvis.utils.make_grid(images_batch, nrow=4)

        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        '''
        plt.show()
