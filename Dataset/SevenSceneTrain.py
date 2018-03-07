from Dataset.SevenScene import SevenScene
from Dataset.mult_modal_transform import ToTensor, ColorJitter, RandomResizedCrop
import re
from torchvision import transforms
import logging


logger = logging.getLogger(__name__)


class SevenSceneTrain(SevenScene):
    def __init__(self, **kwargs):
        self.root_path = kwargs.pop('root_path', None)
        self.transform = kwargs.pop('transform', 'default')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if self.transform == 'default':
            self.transform = transforms.Compose((RandomResizedCrop(224), ColorJitter(), ToTensor()))

        folders = list()
        with open(self.root_path + 'TrainSplit.txt', 'r') as f:
            for line in f:
                fold = 'seq-{:02d}/'.format(int(re.search('(?<=sequence)\d', line).group(0)))
                folders.append(self.root_path + fold)

        logger.info('Loading file name...')
        SevenScene.__init__(self, folders=folders)
        logger.info('Loading finished')

    def __getitem__(self, idx):
        sample = SevenScene.__getitem__(self, idx)
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    from torchvision import utils
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader


    def show_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        grid = utils.make_grid(sample_batched['rgb'])
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    def show_batch_mono(sample_batched):
        """Show image with landmarks for a batch of samples."""
        depth = sample_batched['depth']  # /torch.max(sample_batched['depth'])
        grid = utils.make_grid(depth)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

    tf = transforms.Compose((RandomResizedCrop(224), ColorJitter(), ToTensor()))

    root = '/media/nathan/Data/7_Scenes/chess/'

    dataset = SevenSceneTrain(root_path=root, transform=tf)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    for b in dataloader:
        plt.figure(1)
        show_batch(b)
        plt.figure(2)
        show_batch_mono(b)
        plt.show()
