import setlog
import torch.utils.data as utils
import datasets.custom_quaternion as custom_q
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import re
import pathlib as path
import torchvision as torchvis
import datasets.multmodtf as tf
import matplotlib.pyplot as plt
import torch.utils.data as data
import tqdm
import os
PIL.PngImagePlugin.logger.setLevel('INFO')


logger = setlog.get_logger(__name__)

def rot_to_quat(m):
    if m[2, 2] < 0:
        if m[0, 0] > m[1, 1]:
            t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
            q = np.array([m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]], dtype=np.float32)
        else:
            t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
            q = np.array([m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]], dtype=np.float32)
    else:
        if m[0, 0].item() < -m[1, 1].item():
            t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
            q = np.array([m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t], dtype=np.float32)
        else:
            t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
            q = np.array([t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]], dtype=np.float32)

    q = q * 0.5 / np.sqrt(t)
    return q


def matrix_2_quaternion(mat):
    pos = np.array(mat[0:3, 3], dtype=np.float32)
    rot = np.array(mat[0:3, 0:3], dtype=np.float32)
    '''
    quat = custom_q.Quaternion(matrix=rot)
    quat = quat.q / np.linalg.norm(quat.q)  # Renormalization
    quat = np.array(quat, dtype=np.float32)
    '''
    quat = rot_to_quat(rot)
    return {'position': pos, 'orientation': quat, 'T': mat}


class Base(utils.Dataset):
    def __init__(self, **kwargs):

        self.root_path = kwargs.pop('root', None)
        self.folders = kwargs.pop('folders', None)
        self.depth_factor = kwargs.pop('depth_factor', 1e-3)  # Depth in meter
        self.error_value = kwargs.pop('error_value', 65535)
        self.pose_tf = kwargs.pop('pose_tf', matrix_2_quaternion)
        self.transform = kwargs.pop('transform', None)
        self.used_mod = kwargs.pop('used_mod', ('rgb', 'depth'))

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

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod]})[mod]

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


class Train(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.RandomResizedCrop(224),),
            'rgb': (tf.ColorJitter(), tf.ToTensor()),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)

        self.folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                self.folders.append(self.root_path + fold)

        self.load_data()


class Test(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.Resize((224, 224)),),
            'rgb': (tf.ToTensor(),),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)

        self.folders = list()
        with open(self.root_path + 'TestSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                self.folders.append(self.root_path + fold)

        self.load_data()


class Val(Base):
    def __init__(self, **kwargs):
        default_tf = {
            'first': (tf.Resize((224, 224)),),
            'rgb': (tf.ToTensor(),),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
        pruning = kwargs.pop('pruning', 0.9)
        Base.__init__(self,
                      transform=kwargs.pop('transform', default_tf),
                      **kwargs)
        self.folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                self.folders.append(self.root_path + fold)

        self.load_data()

        step = round(1 / (1-pruning))
        logger.info('Computed step for pruning: {}'.format(step))
        self.data = [dat for i, dat in enumerate(self.data) if i % step == 0]


def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    grid = torchvis.utils.make_grid(sample_batched['rgb'])
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def show_batch_mono(sample_batched):
    """Show image with landmarks for a batch of samples."""
    depth = sample_batched['depth']  # /torch.max(sample_batched['depth'])
    grid = torchvis.utils.make_grid(depth)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':

    logger.setLevel('INFO')
    test_tf = {
            'first': (tf.Resize(240),),
            'rgb': (tf.Equalize(), tf.ToTensor()),
            'depth': (tf.ToTensor(), tf.DepthTransform())
        }
    test_tf_wo_tf = {
            'first': (tf.Resize(240),),
            'rgb': (tf.ToTensor(),),
        }
    root = os.environ['SEVENSCENES'] + 'chess/'

    train_dataset = Train(root=root,
                          transform=test_tf,
                          depth_factor=1e-3)

    train_dataset_wo_tf = Train(root=root,
                                transform=test_tf_wo_tf,
                                used_mod=('rgb',))
    test_dataset = Test(root=root)
    val_dataset = Val(root=root)

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(val_dataset))

    dataloader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    dataloader_wo_tf = data.DataLoader(train_dataset_wo_tf, batch_size=8, shuffle=False, num_workers=2)
    plt.figure(1)
    tmp_batch = dataloader.__iter__().__next__()
    show_batch(tmp_batch)
    plt.figure(2)
    tmp_batch = dataloader_wo_tf.__iter__().__next__()
    show_batch(tmp_batch)
    plt.show()

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        plt.figure(2)
        show_batch_mono(b)
        print(b['pose'])
        plt.show()
