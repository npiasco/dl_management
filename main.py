import setlog

file = 'logging.yaml'
root = '/home/nathan/Dev/Code/dl_management/'
setlog.reconfigure(file, root)

import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
import datasets.multmodtf as tf
import datasets.SevenScene


logger = setlog.get_logger(__name__)


def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    grid = torchvision.utils.make_grid(sample_batched['rgb'])
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def show_batch_mono(sample_batched):
    """Show image with landmarks for a batch of samples."""
    depth = sample_batched['depth']  # /torch.max(sample_batched['depth'])
    grid = torchvision.utils.make_grid(depth)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == '__main__':

    logger.debug('Beginning main')
    logger.info('Root logging')

    tf = torchvision.transforms.Compose((tf.RandomResizedCrop(224), tf.ColorJitter(), tf.ToTensor()))

    root = '/media/nathan/Data/7_Scenes/chess/'
    # root = '/private/anakim/data/7_scenes/chess/'

    dataset = datasets.SevenScene.SevenSceneTrain(root_path=root, transform=tf)

    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        plt.figure(2)
        show_batch_mono(b)
        plt.show()
        break
