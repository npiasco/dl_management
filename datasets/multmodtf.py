import torchvision.transforms.functional as func
import PIL.Image
import torchvision.transforms as tf
import setlog


logger = setlog.get_logger(__name__)


class RandomCrop(tf.RandomCrop):
    def __init__(self, size, padding=0):
        tf.RandomCrop.__init__(self, size=size, padding=padding)

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


class ToTensor(tf.ToTensor):
    def __init__(self, error_value=65535):
        tf.ToTensor.__init__(self)
        self.error_value = error_value
        logger.info('Value error in the depth map set to {}'.format(error_value))

    def __call__(self, sample):
        sample['rgb'] = func.to_tensor(sample['rgb'])
        sample['depth'] = func.to_tensor(sample['depth']).float()
        sample['depth'][sample['depth'] == self.error_value] = 0
        return sample


class ColorJitter(tf.ColorJitter):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05):
        tf.ColorJitter.__init__(self, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        logger.info('Param for color jitter:')
        logger.info('brightness={}, contrast={}, saturation={}, hue={}'.format(brightness, contrast, saturation, hue))

    def __call__(self, sample):
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['rgb'] = transform(sample['rgb'])
        return sample


class Resize(tf.Resize):
    def __init__(self, size):
        tf.Resize.__init__(self, size)

    def __call__(self, sample):
        sample['rgb'] = func.resize(sample['rgb'], self.size, self.interpolation)
        sample['depth'] = func.resize(sample['depth'], self.size, self.interpolation)
        return sample


class RandomResizedCrop(tf.RandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=PIL.Image.BILINEAR):
        tf.RandomResizedCrop.__init__(self, size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def __call__(self, sample):
        img = sample['rgb']
        depth = sample['depth']

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        sample['rgb'] = func.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        sample['depth'] = func.resized_crop(depth, i, j, h, w, self.size, self.interpolation)
        return sample
