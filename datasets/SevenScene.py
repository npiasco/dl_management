import time
import setlog
import torch.utils.data as utils
import datasets.custom_quaternion as custom_q
import pose_utils.utils as putils
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import re
import pathlib as path
import torchvision as torchvis
import torch
import datasets.multmodtf as tf
import matplotlib.pyplot as plt
import torch.utils.data as data
import tqdm
import os
import random
import datasets.augmentation as aug
import copy
PIL.PngImagePlugin.logger.setLevel('INFO')


logger = setlog.get_logger(__name__)


def rot_to_quat(m):
    return putils.rot_to_quat(torch.tensor(m)).numpy()

def matrix_2_quaternion(mat):
    pos = np.array(mat[0:3, 3], dtype=np.float32)
    rot = np.array(mat[0:3, 0:3], dtype=np.float32)
    '''
    quat = custom_q.Quaternion(matrix=rot)
    quat = quat.q / np.linalg.norm(quat.q)  # Renormalization
    quat = np.array(quat, dtype=np.float32)
    '''
    quat = rot_to_quat(rot)

    return {'p': pos, 'q': quat, 'T': mat}


class Base(utils.Dataset):
    def __init__(self, **kwargs):

        self.root_path = kwargs.pop('root', None)
        self.folders = kwargs.pop('folders', None)
        self.error_value = kwargs.pop('error_value', 65535)
        self.pose_tf = kwargs.pop('pose_tf', matrix_2_quaternion)
        self.transform = kwargs.pop('transform', None)
        self.used_mod = kwargs.pop('used_mod', ('rgb', 'depth'))
        self.K = kwargs.pop('K', [[585, 0, 320], [0.0, 585, 240], [0.0, 0.0, 1.0]])

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        self.data = list()

    def load_data(self):
        logger.info('Loading file names')
        for i, folder in tqdm.tqdm(enumerate(self.folders)):
            p = path.Path(folder)
            self.data += [(i, re.search('(?<=-)\d+', file.name).group(0))
                          for file in p.iterdir()
                          if file.is_file() and '.txt' in file.name]
        logger.info('Loading finished')

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        fold, num = self.data[idx]
        sample = dict()
        if 'rgb' in self.used_mod:
            img_name = self.folders[fold] + 'frame-' + num + '.color.png'
            sample['rgb'] = PIL.Image.open(img_name)

        if 'depth' in self.used_mod:
            img_name = self.folders[fold] + 'frame-' + num + '.depth.png'
            sample['depth'] = PIL.Image.open(img_name)

        sample['K'] = np.array(self.K, dtype=np.float32)

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod],
                                                                                    'K': sample['K']})[mod]

        pose_file = self.folders[fold] + 'frame-' + num + '.pose.txt'
        pose = np.ndarray((4, 4), dtype=np.float32)
        with open(pose_file, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        pass

        if self.pose_tf:
            pose = self.pose_tf(pose)

        sample['pose'] = pose

        return sample

    def get_position(self, idx):
        fold, num = self.data[idx]
        pose_file = self.folders[fold] + 'frame-' + num + '.pose.txt'
        pose = np.ndarray((4, 4), dtype=np.float32)
        with open(pose_file, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        pass
        return pose[:3, 3]

class MultiDataset(utils.Dataset):
    def __init__(self, **kwargs):
        utils.Dataset.__init__(self)
        root = kwargs.pop('root', '')
        folders = kwargs.pop('folders', list())
        type = kwargs.pop('type', 'train')
        transform = kwargs.pop('transform', dict())
        aug_transform = kwargs.pop('aug_transform', dict())
        general_options = kwargs.pop('general_options', dict())

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if type == 'train':
            self.datasets = [Train(root=root + folder, transform=transform, **general_options) for folder in folders]
        if type == 'seq':
            self.datasets = [TrainSequence(root=root + folder, transform=transform, **general_options) for folder in
                             folders]
        elif type == 'aug_train':
            self.datasets = [Train(root=root + folder, transform=transform, **general_options) for folder in folders]
            self.datasets += [AugmentedTrain(root=root + folder, transform=transform, **general_options)
                              for folder in folders]
        elif type == 'val':
            self.datasets = [Val(root=root + folder, transform=transform,  **general_options) for folder in folders]
        elif type == 'test':
            self.datasets = [Test(root=root + folder, transform=transform,  **general_options) for folder in folders]
        else:
            raise AttributeError('No implementation for type {}'.format(type))

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        n_dataset = 0
        goon = True
        size_sum_dataset = 0
        while goon:
            if idx >= size_sum_dataset +  len(self.datasets[n_dataset]):
                size_sum_dataset += len(self.datasets[n_dataset])
                n_dataset += 1
            else:
                goon = False

        return self.datasets[n_dataset].__getitem__(idx-size_sum_dataset)

    @property
    def used_mod(self):
        return self.datasets[0].used_mod

    @used_mod.setter
    def used_mod(self, mode):
        for dataset in self.datasets:
            dataset.used_mod = mode


class Train(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.RandomResizedCrop(224),),
            'rgb': (tf.ColorJitter(), tf.ToTensor()),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
        pruning = kwargs.pop('pruning', 0.9)
        sparse_pruning = kwargs.pop('sparse_pruning', False)
        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)

        self.folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                self.folders.append(self.root_path + fold)

        self.load_data()

        if sparse_pruning:
            step = round(1 / (1-pruning))
            logger.info('Computed step for pruning: {}'.format(step))
            self.data = [dat for i, dat in enumerate(self.data) if i % step != 0]
        else:
            self.data = self.data[:round(len(self.data)*pruning)]

class TrainSequence(Train):
    def __init__(self, **kwargs):

        self.num_samples = kwargs.pop('num_samples', 2)
        self.spacing = kwargs.pop('spacing', 20)
        self.random = kwargs.pop('random', True)
        load_fast = kwargs.pop('load_fast', True)

        Train.__init__(self, **kwargs)

        self.poses = list()
        for fold, seq_num in self.data:
            pose_file = self.folders[fold] + 'frame-' + seq_num + '.pose.txt'
            pose = np.ndarray((4, 4), dtype=np.float32)
            with open(pose_file, 'r') as pose_file_pt:
                for i, line in enumerate(pose_file_pt):
                    for j, c in enumerate(line.split('\t')):
                        try:
                            pose[i, j] = float(c)
                        except ValueError:
                            pass
            self.poses.append(pose)

        logger.info('Relative pose computation')
        eye_mat = np.eye(4, 4)
        if not load_fast:
            self.r_poses = [np.argsort([np.linalg.norm(eye_mat - np.matmul(pose, np.linalg.inv(InvnpT)))
                                        for pose in self.poses])
                            for InvnpT in self.poses]

    def __getitem__(self, idx):

        if not hasattr(self, 'r_poses'):
            T = self.poses[idx]
            InvnpT = np.linalg.inv(T)
            eye_mat = np.eye(4, 4)
            d_poses = [np.linalg.norm(eye_mat - np.matmul(pose, InvnpT)) for pose in self.poses]
            nearest_idx = np.argsort(d_poses)
        else:
            nearest_idx = copy.deepcopy(self.r_poses[idx])

        if self.random:
            nearest_idx = nearest_idx[1:(self.num_samples * self.spacing)]
            indexor = torch.randperm(self.num_samples * self.spacing-1).numpy()
            nearest_idx = nearest_idx[indexor]
            nearest_idx = np.concatenate([[idx], nearest_idx])

        samples = list()
        for i in range(0, self.num_samples * self.spacing, self.spacing):
            samples.append(Train.__getitem__(self, nearest_idx[i]))

        return samples


class AugmentedTrain(Train):
    def __init__(self, **kwargs):
        self.zoom_percentage = kwargs.pop('zoom_percentage', 0.2)
        self.tilte_angle = kwargs.pop('tilte_angle', 3.1415 / 16)
        self.reduce_fact = kwargs.pop('reduce_fact', 480/224)
        self.final_depth_size = kwargs.pop('final_depth_size', 56)
        Train.__init__(self, **kwargs)

        self.transform = copy.deepcopy(self.transform)
        self.transform['first'] = (tf.CenterCrop(480), )
        self.transform['depth'] = (tf.ToTensor(), tf.DepthTransform())

    def __getitem__(self, idx):
        sample = Train.__getitem__(self, idx)
        new_sample = aug.creat_new_sample(sample,
                                          zoom=self.zoom_percentage,
                                          reduce_fact=self.reduce_fact,
                                          tilte_angle=self.tilte_angle,
                                          final_size_depth_map=self.final_depth_size)
        return new_sample


class Test(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.Resize((224, 224)),),
            'rgb': (tf.ToTensor(),),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
        light = kwargs.pop('light', False)
        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)

        self.folders = list()
        with open(self.root_path + 'TestSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                self.folders.append(self.root_path + fold)

        self.load_data()

        if light:
            step = 10
            self.data = [dat for i, dat in enumerate(self.data) if i % step == 0]


class Val(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.Resize((224, 224)),),
            'rgb': (tf.ToTensor(),),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
        pruning = kwargs.pop('pruning', 0.9)
        sparse_pruning = kwargs.pop('sparse_pruning', False)

        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)
        self.folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                self.folders.append(self.root_path + fold)

        self.load_data()

        if sparse_pruning:
            step = round(1 / (1-pruning))
            logger.info('Computed step for pruning: {}'.format(step))
            self.data = [dat for i, dat in enumerate(self.data) if i % step == 0]
        else:
            self.data = self.data[round(len(self.data)*pruning):]


def show_batch(sample_batched,  n_row=4):
    """Show image with landmarks for a batch of samples."""
    grid = torchvis.utils.make_grid(sample_batched['rgb'], nrow=n_row)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def show_seq_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    grid = torchvis.utils.make_grid(torch.cat([batched['rgb'] for batched in sample_batched]), nrow=2)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def show_batch_mono(sample_batched, n_row=4):
    """Show image with landmarks for a batch of samples."""
    depth = sample_batched['depth']  # /torch.max(sample_batched['depth'])
    grid = torchvis.utils.make_grid(depth, nrow=n_row)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':

    logger.setLevel('INFO')
    test_tf = {
            'first': (tf.Resize(256), tf.CenterCrop(256), ),
            'rgb': (tf.ToTensor(), ),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
    test_tf_wo_tf = {
            'first': (tf.Resize(240),),
            'rgb': (tf.ToTensor(),),
        }
    root = os.environ['SEVENSCENES'] + 'heads/'
    '''
    train_dataset = Train(root=root,
                          transform=test_tf)

    train_dataset_wo_tf = Train(root=root,
                                transform=test_tf_wo_tf,
                                used_mod=('rgb',))
    '''
    test_dataset = Test(root=root, light=True)
    '''
    val_dataset = Val(root=root)

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(val_dataset))
    '''
    train_seq_dataset = TrainSequence(root=root, transform=test_tf,
                                      num_samples=2, spacing=10, random=True)
    val_seq_dataset = Val(root=root,
                              transform=test_tf,
                              )

    mult_train_seq_dataset = MultiDataset(type='val', root=os.environ['SEVENSCENES'],
                                          folders=['fire/', 'heads/'], transform=test_tf,
                                          general_options={'used_mod': ('rgb',)})

    mult_train_seq_dataset.used_mod = ('depth',)

    #dataloader = data.DataLoader(mult_train_seq_dataset, batch_size=16, shuffle=False, num_workers=8)
    dataloader = data.DataLoader(train_seq_dataset, batch_size=1, shuffle=False, num_workers=0)
    '''
    dataloader_wo_tf = data.DataLoader(train_dataset_wo_tf, batch_size=8, shuffle=False, num_workers=2)
    plt.figure(1)
    tmp_batch = dataloader.__iter__().__next__()
    show_batch(tmp_batch)
    plt.figure(2)
    tmp_batch = dataloader_wo_tf.__iter__().__next__()
    show_batch(tmp_batch)
    plt.show()
    '''
    plt.ion()
    plt.show()
    fig = plt.figure(1)
    print(len(train_seq_dataset))
    print(len(val_seq_dataset))
    print(len(test_dataset))
    for i, b in enumerate(dataloader):
        #show_seq_batch(b)
        fig.clear()
        show_seq_batch(b)
        plt.pause(0.75)
        '''
        fig.clear()
        show_batch_mono(b)
        print(i)
        plt.pause(0.2)
        '''
        del b
