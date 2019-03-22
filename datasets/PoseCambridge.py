import setlog
import torch.utils.data as utils
import pose_utils.utils as putils
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import pandas as pd
import torchvision as torchvis
import torch
import datasets.multmodtf as tf
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
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
        self.folders = kwargs.pop('folders', '')
        self.error_value = kwargs.pop('error_value', 65535)
        self.pose_tf = kwargs.pop('pose_tf', matrix_2_quaternion)
        self.transform = kwargs.pop('transform', None)
        self.used_mod = kwargs.pop('used_mod', ('rgb', ))
        self.K = kwargs.pop('K', [[2179.2, 0, 960], [0.0, 2179.2, 540], [0.0, 0.0, 1.0]])

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        self.data = list()

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        resource = self.data[idx]
        sample = dict()

        img_name = self.root_path + self.folders + resource[0]
        sample['rgb'] = PIL.Image.open(img_name)

        sample['K'] = np.array(self.K, dtype=np.float32)

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod],
                                                                                    'K': sample['K']})[mod]

        t = resource[1:4].astype('float') # m
        q = resource[4:8].astype('float')
        pose = np.zeros((4, 4), dtype=np.float32)
        R = putils.quat_to_rot(torch.tensor(q)).t().numpy()
        pose[:3, :3] = R
        pose[:3, 3] = t
        pose[3, 3] = 1
        q = putils.rot_to_quat(torch.from_numpy(pose[:3, :3])).numpy()
        sample['pose'] = {'p': t, 'q': q, 'T': pose}

        return sample

    def get_position(self, idx):
        resource = self.data[idx]
        t = resource[1:4].astype('float') # m
        return t

class Train(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.RandomResizedCrop(224),),
            'rgb': (tf.ToTensor()),
        }
        pruning = kwargs.pop('pruning', 0.9)
        sparse_pruning = kwargs.pop('sparse_pruning', False)
        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)

        data_file_name = self.root_path + self.folders + 'dataset_train.txt'
        self.data = pd.read_csv(data_file_name, header=1, sep=' ').values

        if sparse_pruning:
            step = round(1 / (1-pruning))
            logger.info('Computed step for pruning: {}'.format(step))
            indexor = np.zeros(len(self.data))
            for i in range(len(self.data)):
                if i % step != 0:
                    indexor[i] = 1
            self.data = self.data[indexor.astype(bool)]
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
        for da in self.data:
            t = da[1:4].astype('float')
            q = da[4:8].astype('float')
            pose = np.zeros((4, 4), dtype=np.float32)
            R = putils.quat_to_rot(torch.tensor(q)).numpy()
            pose[:3, :3] = R
            pose[:3, 3] = t
            pose[3, 3] = 1

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

        data_file_name = self.root_path + 'dataset_test.txt'
        self.data = pd.read_csv(data_file_name, header=1, sep=' ').values

        if light:
            step = 10
            indexor = np.zeros(len(self.data))
            for i in range(len(self.data)):
                if i % step == 0:
                    indexor[i] = 1

            self.data = self.data[indexor.astype(bool)]


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

        data_file_name = self.root_path + 'dataset_train.txt'
        self.data = pd.read_csv(data_file_name, header=1, sep=' ').values

        if sparse_pruning:
            step = round(1 / (1-pruning))
            logger.info('Computed step for pruning: {}'.format(step))
            indexor = np.zeros(len(self.data))
            for i in range(len(self.data)):
                if i % step == 0:
                    indexor[i] = 1
            self.data = self.data[indexor]
        else:
            self.data = self.data[round(len(self.data)*pruning):]


class MultiDataset(utils.Dataset):
    def __init__(self, **kwargs):
        utils.Dataset.__init__(self)
        root = kwargs.pop('root', '')
        folders = kwargs.pop('folders', list())
        type = kwargs.pop('type', 'train')
        transform = kwargs.pop('transform', dict())
        general_options = kwargs.pop('general_options', dict())

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if type == 'train':
            self.datasets = [Train(root=root + folder, transform=transform, **general_options) for folder in folders]
        if type == 'seq':
            self.datasets = [TrainSequence(root=root + folder, transform=transform, **general_options) for folder in
                             folders]
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
            'first': (tf.Resize(140), tf.RandomCrop((112, 224))),
            'rgb': (tf.ToTensor(), ),
        }
    test_tf_wo_tf = {
            'first': (tf.Resize(240),),
            'rgb': (tf.ToTensor(),),
        }
    root = os.environ['CAMBRIDGE']
    train_dataset = TrainSequence(root=root, folders='Street/',
                                  transform=test_tf, spacing=1, num_samples=8, random=False)

    print(len(train_dataset))

    dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)
    plt.figure(1)

    plt.ion()
    plt.show()
    fig = plt.figure(1)

    for i, b in enumerate(dataloader):
        #show_seq_batch(b)
        fig.clear()
        show_seq_batch(b)
        #show_batch(b)
        plt.pause(2)


        del b
