import logging
import torch.utils.data as utils
import datasets.custom_quaternion as custom_q
import numpy as np
import PIL
import re
import pathlib as path
import torchvision as torchvis
import datasets.mult_modal_transform as tf


sslogger = logging.getLogger('main.SS')

def matrix_2_quaternion(mat):
    pos = np.array(mat[0:3, 3])
    rot = np.array(mat[0:3, 0:3])
    quat = custom_q.Quaternion(matrix=rot)
    quat = quat.q / np.linalg.norm(quat.q)  # Renormalization
    return {'position': pos, 'orientation': quat}


class SevenScene(utils.Dataset):

    def __init__(self, **kwargs):
        #self.logger = logging.getLogger('qualcity.'+__name__+'.SevenSceneClass')
#        self.logger = logging.getLogger('main.SS.SevenSceneClass')

        self.folders = kwargs.pop('folders', None)
        self.depth_factor = kwargs.pop('depth_factor', 1e-3)  # Depth in meter
        self.pose_tf = kwargs.pop('pose_tf', matrix_2_quaternion)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        self.data = list()
       # self.logger.info('Loading file name...')
        for i, folder in enumerate(self.folders):
            p = path.Path(folder)
            self.data += [(i, re.search('(?<=-)\d+', file.name).group(0))
                          for file in p.iterdir()
                          if file.is_file() and '.txt' in file.name]
        #self.logger.info('Loading finished')


    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        fold, num = self.data[idx]
        img_name = self.folders[fold] + 'frame-' + num + '.color.png'
        rgb = PIL.Image.open(img_name)

        img_name = self.folders[fold] + 'frame-' + num + '.depth.png'
        depth = PIL.Image.open(img_name)

        pose_file = self.folders[fold] + 'frame-' + num + '.pose.txt'
        pose = np.ndarray((4, 4), dtype=float)
        with open(pose_file, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        #self.logger.warning('Error reading pose file')
                        pass

        if self.pose_tf:
            pose = self.pose_tf(pose)
        sample = {'rgb': rgb, 'depth': depth, 'pose': pose}

        return sample


class SevenSceneTrain(SevenScene):
    def __init__(self, **kwargs):
        self.root_path = kwargs.pop('root_path', None)
        self.transform = kwargs.pop('transform', 'default')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = torchvis.transforms.Compose((tf.RandomResizedCrop(224), tf.ColorJitter(), tf.ToTensor()))

        folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                folders.append(self.root_path + fold)

        SevenScene.__init__(self, folders=folders)


    def __getitem__(self, idx):
        sample = SevenScene.__getitem__(self, idx)
        if self.transform:
            sample = self.transform(sample)
        return sample


class SevenSceneTest(SevenScene):
    def __init__(self, **kwargs):
        self.root_path = kwargs.pop('root_path', None)
        self.transform = kwargs.pop('transform', 'default')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = torchvis.transforms.Compose((tf.Resize((224, 224)), tf.ToTensor()))

        folders = list()
        with open(self.root_path + 'TestSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                folders.append(self.root_path + fold)

        SevenScene.__init__(self, folders=folders)

    def __getitem__(self, idx):
        sample = SevenScene.__getitem__(self, idx)
        if self.transform:
            sample = self.transform(sample)
        return sample


class SevenSceneVal(SevenScene):
    def __init__(self, **kwargs):
        self.root_path = kwargs.pop('root_path', None)
        self.transform = kwargs.pop('transform', 'default')
        pruning = kwargs.pop('pruning', 0.9)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = torchvis.transforms.Compose((tf.Resize((224, 224)), tf.ToTensor()))

        folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                folders.append(self.root_path + fold)

        SevenScene.__init__(self, folders=folders)

        step = round(1 / (1-pruning))
        #self.logger.info('Computed step {}'.format(step))
        self.data = [dat for i, dat in enumerate(self.data) if i % step == 0]

    def __getitem__(self, idx):
        sample = SevenScene.__getitem__(self, idx)
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch.utils.data as data
    import yaml
    import time

    def show_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        grid = torchvis.utils.make_grid(sample_batched['rgb'])
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    def show_batch_mono(sample_batched):
        """Show image with landmarks for a batch of samples."""
        depth = sample_batched['depth']  # /torch.max(sample_batched['depth'])
        grid = torchvis.utils.make_grid(depth)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    tf = torchvis.transforms.Compose((tf.RandomResizedCrop(224), tf.ColorJitter(), tf.ToTensor()))

    root = '/media/nathan/Data/7_Scenes/chess/'

    dataset = SevenSceneTrain(root_path=root, transform=tf)
    print(len(dataset))

    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        plt.figure(2)
        show_batch_mono(b)
        plt.show()

