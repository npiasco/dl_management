from torch.utils.data import Dataset
from Dataset.custom_quaternion import Quaternion
import numpy as np
from PIL import Image
import re
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


def matrix_2_quaternion(mat):
    pos = np.array(mat[0:3, 3])
    rot = np.array(mat[0:3, 0:3])
    quat = Quaternion(matrix=rot)
    quat = quat.q / np.linalg.norm(quat.q)  # Renormalization
    return {'position': pos, 'orientation': quat}


class SevenScene(Dataset):

    def __init__(self, **kwargs):
        self.folders = kwargs.pop('folders', None)
        self.depth_factor = kwargs.pop('depth_factor', 1e-3)  # Depth in meter
        self.pose_tf = kwargs.pop('pose_tf', matrix_2_quaternion)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)
        self.data = list()
        for i, folder in enumerate(self.folders):
            p = Path(folder)
            self.data += [(i, re.search('(?<=-)\d+', file.name).group(0))
                          for file in p.iterdir()
                          if file.is_file() and '.txt' in file.name]

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        fold, num = self.data[idx]
        img_name = self.folders[fold] + 'frame-' + num + '.color.png'
        rgb = Image.open(img_name)

        img_name = self.folders[fold] + 'frame-' + num + '.depth.png'
        depth = Image.open(img_name)

        pose_file = self.folders[fold] + 'frame-' + num + '.pose.txt'
        pose = np.ndarray((4, 4), dtype=float)
        with open(pose_file, 'r') as pose_file_pt:
            for i, line in enumerate(pose_file_pt):
                for j, c in enumerate(line.split('\t')):
                    try:
                        pose[i, j] = float(c)
                    except ValueError:
                        logger.warning('Error reading pose file')
                        pass

        if self.pose_tf:
            pose = self.pose_tf(pose)
        sample = {'rgb': rgb, 'depth': depth, 'pose': pose}

        return sample


if __name__ == '__main__':
    pass
