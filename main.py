import yaml
import logging.config
import logging
import time
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as data
import datasets.mult_modal_transform as tf



def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    grid = torchvision.utils.make_grid(sample_batched['rgb'])
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def show_batch_mono(sample_batched):
    """Show image with landmarks for a batch of samples."""
    depth = sample_batched['depth']  # /torch.max(sample_batched['depth'])
    grid = torchvision.utils.make_grid(depth)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


def config_log():
    root = '/home/nathan/Dev/Code/dl_management/'
    path = root + 'logging.yaml'
    with open(path, 'rt') as f:
        config = yaml.safe_load(f.read())
    config['handlers']['file']['filename'] = root + '.log/run_{}.log'.format(time.time())
    print(config)
    return config


if __name__ == '__main__':
    config = config_log()

    logging.config.dictConfig(config)
    logger = logging.getLogger('main')

    logger.debug(config)
    logger.info('Root logging')
    import datasets.SevenScene as ssdataset

    tf = torchvision.transforms.Compose((tf.RandomResizedCrop(224), tf.ColorJitter(), tf.ToTensor()))

    root = '/media/nathan/Data/7_Scenes/chess/'

    dataset = ssdataset.SevenSceneTrain(root_path=root, transform=tf)
    print(len(dataset))

    dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        plt.figure(2)
        show_batch_mono(b)
        plt.show()