import torchvision.transforms.functional as func
import PIL.Image
import torchvision.transforms as tf
import setlog


logger = setlog.get_logger(__name__)


class RandomCrop(tf.RandomCrop):
    def __init__(self, size, padding=0):
        tf.RandomCrop.__init__(self, size=size, padding=padding)

    def __call__(self, sample):
        for name, mod in sample.items():
            if self.padding > 0:
                sample[name] = func.pad(mod, self.padding)

        first_mod = list(sample.values())[0]
        i, j, h, w = self.get_params(first_mod, self.size)

        for name, mod in sample.items():
            sample[name] = func.crop(mod, i, j, h, w)
        return sample


class ToTensor(tf.ToTensor):
    def __init__(self):
        tf.ToTensor.__init__(self)

    def __call__(self, sample):
        for name, mod in sample.items():
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
            sample[name] = transform(mod)

        return sample


class Resize(tf.Resize):
    def __init__(self, size):
        tf.Resize.__init__(self, size)

    def __call__(self, sample):
        for name, mod in sample.items():
            sample[name] = func.resize(mod, self.size, self.interpolation)

        return sample


class RandomResizedCrop(tf.RandomResizedCrop):
    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=PIL.Image.BILINEAR):
        tf.RandomResizedCrop.__init__(self, size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def __call__(self, sample):
        first_mod = list(sample.values())[0]

        i, j, h, w = self.get_params(first_mod, self.scale, self.ratio)

        for name, mod in sample.items():
            sample[name] = func.resized_crop(mod, i, j, h, w, self.size, self.interpolation)

        return sample


class DepthTransform:
    def __init__(self, depth_factor=1e-3, error_value=65535, replacing_value=0):
        self.depth_factor = depth_factor
        self.error_value = error_value
        self.replacing_value = replacing_value

    def __call__(self, sample):
        for name, mod in sample.items():
            sample[name][sample[name] == self.error_value] = self.replacing_value
            sample[name] *= self.depth_factor

        return sample


class Normalize(tf.Normalize):
    def __init__(self, mean, std):
        tf.Normalize.__init__(self, mean, std)

    def __call__(self, sample):
        for name, mod in sample.items():
            sample[name] = func.normalize(mod, self.mean, self.std)

        return sample
