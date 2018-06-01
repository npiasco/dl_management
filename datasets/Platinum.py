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
import scipy.misc


logger = setlog.get_logger(__name__)


class Platinum(utils.data.Dataset):
    def __init__(self, root, file, modalities, **kwargs):
        self.root = root
        self.transform = kwargs.pop('transform', 'default')
        self.bearing = kwargs.pop('bearing', True)
        self.panorama_split = kwargs.pop('panorama_split', {'v_split': 3,
                                                            'h_split': 2,
                                                            'offset': 0})

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = {
                'first': (tf.Resize((224, 224)), tf.ToTensor())
            }

        self.data = pd.read_csv(self.root + file, header=1, sep=',')
        self.modalities = modalities
        self.used_mod = self.modalities

    def __len__(self):
        s = self.data.__len__()
        if self.panorama_split is not None:
            s *= self.panorama_split['h_split'] * self.panorama_split['v_split']
        return s

    def __getitem__(self, idx):
        sample = dict()
        for mod_name in self.used_mod:
            if self.panorama_split is not None:
                split_idx = idx % (self.panorama_split['h_split'] * self.panorama_split['v_split'])
                fidx = idx // (self.panorama_split['h_split'] * self.panorama_split['v_split'])
            else:
                fidx = idx
            file_name = self.root + self.data.ix[fidx, self.mod_to_indx(mod_name)] + '.png'
            if self.panorama_split is not None:
                raw_img = scipy.misc.imread(file_name)
                r = raw_img.shape[1] / (2 * np.pi)
                vert_ang = np.degrees(raw_img.shape[0]/r)
                v_pas_angle = (360 / self.panorama_split['v_split'])
                h_pas_angle = (vert_ang / self.panorama_split['h_split'])
                offset = self.panorama_split['offset']

                v_im_num = split_idx % self.panorama_split['v_split']
                h_im_num = split_idx // self.panorama_split['v_split']

                size_im = [int(2 * np.tan(np.radians(h_pas_angle/2)) * r),
                           int(2 * np.tan(np.radians(v_pas_angle/2)) * r),
                           3]
                color_img = np.zeros((size_im[0], size_im[1], size_im[2]), np.uint8)
                for j in range(size_im[0]):
                    for i in range(size_im[1]):
                        if i >= size_im[1]:
                            x_angle = np.radians(offset + v_im_num * v_pas_angle + v_pas_angle/2) + \
                                      np.arctan((i - size_im[1]/2)/r)
                        else:
                            x_angle = np.radians(offset + v_im_num * v_pas_angle + v_pas_angle/2) - \
                                      np.arctan((size_im[1]/2 - i)/r)
                        if j >= size_im[0]:
                            y_angle = np.radians(h_im_num * h_pas_angle + h_pas_angle / 2) + \
                                      np.arctan((j - size_im[0]/2)/r)
                        else:
                            y_angle = np.radians(h_im_num * h_pas_angle + h_pas_angle / 2) - \
                                      np.arctan((size_im[0]/2 - j)/r)
                        im_j = int(r * y_angle)
                        im_i = int(r * x_angle)
                        color_img[j, i, :] = raw_img[im_j, im_i, :]

                sample[mod_name] = PIL.Image.fromarray(color_img)
            else:
                sample[mod_name] = PIL.Image.open(file_name)

        if self.transform:
            if 'first' in self.transform:
                sample = torchvis.transforms.Compose(self.transform['first'])(sample)
            for mod in self.transform:
                if mod not in ('first',) and mod in self.used_mod:
                    sample[mod] = torchvis.transforms.Compose(self.transform[mod])({mod: sample[mod]})[mod]

        sample['coord'] = self.data.ix[fidx, 1:3].as_matrix().astype('float')
        return sample

    @staticmethod
    def mod_to_indx(mod):
        return {'rgb': 3, 'depth': 4, 'sem': 5}.get(mod)


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
    grid = torchvis.utils.make_grid(images_batch, nrow=3)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':
    root_to_folders = os.environ['PLATINUM'] + 'data/'

    modtouse = ['rgb', 'depth', 'sem']
    transform = {
        'first': (tf.Resize((256)),tf.RandomCrop(224)),
        'rgb': (tf.ToTensor(), ),
        'depth': (tf.ToTensor(),),
        'sem': (tf.ToTensor(),)
    }

    dataset = Platinum(root=root_to_folders,
                       file='test_graph.csv',
                       modalities=modtouse,
                       transform=transform)

    dataloader = utils.data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        print(b['coord'])
        plt.show()
