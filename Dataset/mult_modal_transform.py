from torchvision.transforms import RandomCrop as OriginalRandomCrop
from torchvision.transforms import ToTensor as OriginalToTensor
from torchvision.transforms import ColorJitter as OriginalColorJitter
from torchvision.transforms import Resize as OriginalResize
from torchvision.transforms import RandomResizedCrop as OriginalRandomResizedCrop
import torchvision.transforms.functional as func
from PIL import Image


class RandomCrop(OriginalRandomCrop):
    def __init__(self, size, padding=0):
        OriginalRandomCrop.__init__(self, size=size, padding=padding)

    def __call__(self, sample):
        img = sample['rgb']
        depth = sample['depth']
        if self.padding > 0:
            img = func.pad(img, self.padding)
            depth = func.pad(depth, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        sample['rgb'] = func.crop(img, i, j, h, w)
        sample['depth'] = func.crop(depth, i, j, h, w)
        return sample


class ToTensor(OriginalToTensor):
    def __init__(self, error_value=65535, depth_factor=1e-3):
        OriginalToTensor.__init__(self)
        self.depth_factor = depth_factor
        self.error_value = error_value

    def __call__(self, sample):
        sample['rgb'] = func.to_tensor(sample['rgb'])
        sample['depth'] = func.to_tensor(sample['depth']).float()
        sample['depth'][sample['depth'] == self.error_value] = 0
        sample['depth'] *= self.depth_factor
        return sample


class ColorJitter(OriginalColorJitter):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        OriginalColorJitter.__init__(self, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['rgb'] = transform(sample['rgb'])
        return sample


class Resize(OriginalResize):
    def __init__(self, size):
        OriginalResize.__init__(self, size)

    def __call__(self, sample):
        sample['rgb'] = func.resize(sample['rgb'], self.size, self.interpolation)
        sample['depth'] = func.resize(sample['depth'], self.size, self.interpolation)
        return sample


class RandomResizedCrop(OriginalRandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        OriginalRandomResizedCrop.__init__(self, size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def __call__(self, sample):
        img = sample['rgb']
        depth = sample['depth']

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        sample['rgb'] = func.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        sample['depth'] = func.resized_crop(depth, i, j, h, w, self.size, self.interpolation)
        return sample
