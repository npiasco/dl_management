import setlog
import torch.utils as utils
import torch.utils.data
import torchvision as torchvis
import datasets.multmodtf as tf
import pandas as pd
import numpy as np
import PIL.Image
import torch
import matplotlib.pyplot as plt

logger = setlog.get_logger(__name__)


class VBLDataset(utils.data.Dataset):

    def __init__(self, root, coord_file, modalities, **kwargs):
        self.root = root
        self.transform = kwargs.pop('transform', 'default')

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = {
                'first': (tf.Resize((224, 224)), tf.ToTensor())
            }

        self.coord = pd.read_csv(self.root + coord_file, header=None, sep='\t', dtype=np.float64)

        self.modalities = dict()
        for mod_name in modalities:
            self.modalities[mod_name] = pd.read_csv(self.root + modalities[mod_name], header=None)

        self.used_mod = self.modalities.keys()

    def __len__(self):
        return len(self.coord)

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

        sample['coord'] = self.coord.ix[idx, 0:1].as_matrix().astype('float')
        return sample


def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    buffer = tuple()
    for name, mod in sample_batched.items():
        if name not in ('coord',):
            buffer += (mod,)

    images_batch = torch.cat(buffer, 0)
    grid = torchvis.utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    root_to_folders = '/media/nathan/Data/Robotcar/training/TrainDataset_02_10_15/'
    modtouse = {'rgb': 'dataset.txt', 'depth': 'depth_dataset.txt', 'ref': 'ref_dataset.txt'}
    transform = {
        'first': (tf.RandomResizedCrop(420),),
        'rgb': (tf.ColorJitter(), tf.ToTensor()),
        'depth': (tf.ToTensor(),),
        'ref': (tf.ToTensor(),)
    }
    dataset = VBLDataset(root=root_to_folders, modalities=modtouse, coord_file='coordxIm.txt', transform=transform)
    dataset.used_mod = ['rgb']
    dataloader = utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        print(b['coord'])
        plt.show()
