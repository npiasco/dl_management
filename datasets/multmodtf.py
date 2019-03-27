import torchvision.transforms.functional as func
import PIL.Image
import PIL.ImageOps
import torchvision.transforms as tf
import setlog
import torch.nn.functional as nn_func
import torch
import matplotlib.pyplot as plt
import random


logger = setlog.get_logger(__name__)


class RandomVerticalFlip(tf.RandomVerticalFlip):
    def __init__(self, p=0.5):
        tf.RandomVerticalFlip.__init__(self, p=p)

    def __call__(self, sample):
        for name, mod in sample.items():
            if name != 'K':
                if random.random() < self.p:
                    sample[name] = func.vflip(mod)

        return sample


class RandomHorizontalFlip(tf.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        tf.RandomHorizontalFlip.__init__(self, p=p)

    def __call__(self, sample):
        for name, mod in sample.items():
            if name != 'K':
                if random.random() < self.p:
                    sample[name] = func.hflip(mod)

        return sample


class RandomCrop(tf.RandomCrop):
    def __init__(self, size, padding=0):
        tf.RandomCrop.__init__(self, size=size, padding=padding)

    def __call__(self, sample):
        for name, mod in sample.items():
            if self.padding > 0:
                sample[name] = func.pad(mod, self.padding)

        first_mod = [mod for name, mod in sample.items() if name is not 'K'][0]
        i, j, h, w = self.get_params(first_mod, self.size)

        for name, mod in sample.items():
            if name is 'K':
                sample['K'][0, 2] -= j
                sample['K'][1, 2] -= i
            else:
                sample[name] = func.crop(mod, i, j, h, w)
        return sample


class CenterCrop(tf.CenterCrop):
    def __init__(self, size):
        tf.CenterCrop.__init__(self, size=size)

    def __call__(self, sample):
        w, h = self.size
        th, tw = [mod for name, mod in sample.items() if name is not 'K'][0].size
        j = round(th - h) / 2.
        i = round(tw - w) / 2.

        for name, mod in sample.items():
            if name is 'K':
                sample['K'][0, 2] -= j
                sample['K'][1, 2] -= i
            else:
                sample[name] = func.center_crop(mod, self.size)
        return sample


class ToTensor(tf.ToTensor):
    def __init__(self):
        tf.ToTensor.__init__(self)

    def __call__(self, sample):
        for name, mod in sample.items():
            if name is not 'K':
                sample[name] = func.to_tensor(mod).float()

        return sample


class ColorJitter(tf.ColorJitter):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        tf.ColorJitter.__init__(self, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        logger.info('Param for color jitter:')
        logger.info('brightness={}, contrast={}, saturation={}, hue={}'.format(brightness, contrast, saturation, hue))

    def __call__(self, sample):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        for name, mod in sample.items():
            if name is not 'K':
                sample[name] = transform(mod)

        return sample


class Resize(tf.Resize):
    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        tf.Resize.__init__(self, size, interpolation=interpolation)

    def __call__(self, sample):
        w, h = [mod for name, mod in sample.items() if name is not 'K'][0].size
        for name, mod in sample.items():
            if name in ['rgb']:
                sample[name] = func.resize(mod, self.size, self.interpolation)
            elif name is 'K':
                if isinstance(self.size, int):
                    if w < h:
                        ratiow = ratioh = self.size / w

                    else:
                        ratiow = ratioh = self.size / h
                else:
                    ratiow = self.size[0] / w
                    ratioh = self.size[1] / h
                sample['K'][0, :] *= ratiow
                sample['K'][1, :] *= ratioh
            else:
                sample[name] = func.resize(mod, self.size, PIL.Image.NEAREST)

        return sample


class ResizeK:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        for name, mod in sample.items():
            if name is not 'K':
                raise AttributeError('Can only apply ResizeK tf on mode K')
            sample[name][:2, :] *= self.ratio
        return sample


class RandomResizedCrop(tf.RandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=PIL.Image.BILINEAR):
        tf.RandomResizedCrop.__init__(self, size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def __call__(self, sample):
        first_mod = list(sample.values())[0]

        i, j, h, w = self.get_params(first_mod, self.scale, self.ratio)

        for name, mod in sample.items():
            if name in ['rgb']:
                sample[name] = func.resized_crop(mod, i, j, h, w, self.size, self.interpolation)
            else:
                sample[name] = func.resized_crop(mod, i, j, h, w, self.size, PIL.Image.NEAREST)

        return sample


class DepthTransform:
    def __init__(self, depth_factor=1e-3, error_value=65535, replacing_value=0, inverse=False):
        self.depth_factor = depth_factor
        self.error_value = error_value
        self.replacing_value = replacing_value
        self.inverse = inverse

    def __call__(self, sample):
        for name, mod in sample.items():
            if name is not 'K':
                sample[name][sample[name] == self.error_value] = self.replacing_value
                sample[name] *= self.depth_factor
                if self.inverse:
                    sample[name] = torch.reciprocal(sample[name] + 1)

        return sample


class UniformPruning:
    def __init__(self, **kwargs):
        self.replacement_value = kwargs.pop('replacement_value', 0.0)
        self.kernel_size = kwargs.pop('kernel_size', 20)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, sample):
        moy_density = None
        for name, mod in sample.items():
            c, w, h = mod.size()
            for i in range(0, w, self.kernel_size):
                for j in range(0, h, self.kernel_size):
                    cropped = self.crop_sample(mod, w, h, i, j)
                    density = torch.numel(torch.nonzero(cropped))/(self.kernel_size**2)
                    if density > 0:
                        if moy_density is None:
                            moy_density = density
                        else:
                            moy_density = (moy_density + density)/2

            if moy_density is None:
                logger.warning('No point found')
                return sample

            for i in range(0, w, self.kernel_size):
                for j in range(0, h, self.kernel_size):
                    cropped = self.crop_sample(mod, w, h, i, j)
                    density = torch.numel(torch.nonzero(cropped)) / (self.kernel_size ** 2)
                    f_density = density/moy_density
                    if f_density > 1.5:
                        self.prune(mod, w, h, i, j, 1/f_density)

            sample[name] = mod

        return sample

    def crop_sample(self, sample, w, h, i, j):
        if i + self.kernel_size > w:
            cropped_x = sample[:, -self.kernel_size:, :]
        else:
            cropped_x = sample[:, i:i + self.kernel_size, :]
        if j + self.kernel_size > h:
            cropped = cropped_x[:, :, -self.kernel_size:]
        else:
            cropped = cropped_x[:, :, j:j + self.kernel_size]

        return cropped

    def prune(self, sample,  w, h, i, j, probalility):
        indexor = torch.rand(self.kernel_size**2) > probalility
        if i + self.kernel_size > w:
            if j + self.kernel_size > h:
                sample[:, -self.kernel_size:, -self.kernel_size:][indexor] = self.replacement_value
            else:
                sample[:, -self.kernel_size:, j:j + self.kernel_size][indexor] = self.replacement_value
        else:
            if j + self.kernel_size > h:
                sample[:, i:i + self.kernel_size, -self.kernel_size:][indexor] = self.replacement_value
            else:
                sample[:, i:i + self.kernel_size, j:j + self.kernel_size][indexor] = self.replacement_value


class Normalize(tf.Normalize):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        tf.Normalize.__init__(self, mean, std)

    def __call__(self, sample):
        for name, mod in sample.items():
            if name is not 'K':
                sample[name] = func.normalize(mod, self.mean, self.std)

        return sample


class Equalize:
    def __init__(self, mask=None):
        self.mask = mask

    def __call__(self, sample):
        for name, mod in sample.items():
            if name is not 'K':
                sample[name] = PIL.ImageOps.equalize(mod, mask=self.mask)

        return sample


class JetTransform:
    def __init__(self, cmap='jet', s_lut=256):
        self.cmap = plt.get_cmap(cmap, lut=s_lut)

    def __call__(self, sample):
        for name, mod in sample.items():
            #mod = mod - torch.min(mod)
            sample[name] = torch.Tensor(
                    self.cmap(mod.numpy()).transpose((0,3,1,2))
            )[:,0:3,:,:].squeeze()

        return sample


class GradNorm:
    def __init__(self):
        self.w_x = torch.autograd.Variable(torch.Tensor([[n * 0.1 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]],
                                                         [n * 0.4 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]],
                                                         [n * 0.6 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]],
                                                         [n * 1.0 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]],
                                                         [n * 0.6 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]],
                                                         [n * 0.4 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]],
                                                         [n * 0.1 for n in [0.5, 1.0, 1.5, 0.0, -1.5, -1.0, -0.5]]
                                                         ]).view(1,1,7,7), requires_grad=False)
        #self.w_x /= torch.norm(self.w_x)
        self.w_x = torch.autograd.Variable(torch.rand(1,1,3,3))
        self.w_y = torch.autograd.Variable(torch.Tensor([[n * 0.1 for n in [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]],
                                                         [n * 0.5 for n in [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]],
                                                         [n * 1.0 for n in [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]],
                                                         [0, 0, 0, 0, 0, 0, 0],
                                                         [n * -1.0 for n in [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]],
                                                         [n * -0.5 for n in [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]],
                                                         [n * -0.1 for n in [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]]
                                                         ]).view(1,1,7,7), requires_grad=False)
        self.w_y = torch.autograd.Variable(torch.rand(1, 1, 3, 3))
        self.w_z = torch.autograd.Variable(torch.rand(1, 1, 3, 3))
        #self.w_y /= torch.norm(self.w_y)

    def __call__(self, sample):
        for name, mod in sample.items():
            mod = torch.autograd.Variable(mod.unsqueeze(dim=0))
            sample[name] = torch.cat(
                (
                    nn_func.conv2d(mod, self.w_x, padding=1, dilation=1),
                    nn_func.conv2d(mod, self.w_y, padding=1, dilation=1),
                    nn_func.conv2d(mod, self.w_z, padding=1, dilation=1),
                    # mod
                ),
                dim = 1
            ).squeeze().data

        return sample
