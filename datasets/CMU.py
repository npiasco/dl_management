import setlog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib as path
import re
import sklearn.neighbors as nn
from mpl_toolkits.mplot3d import axes3d


logger = setlog.get_logger(__name__)


def in_range_2pi(angle):
    if angle>2*np.pi:
        return angle - 2*np.pi
    elif angle<0:
        return 2 * np.pi + angle
    else:
        return angle


def prune_path(path, tolerance=1):
    def sdistance_pts(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    idx = np.ones(path.shape[0])

    cursor = 0
    squarred_t = tolerance**2

    for i, point in enumerate(path[1:, :]):
        if sdistance_pts(point, path[cursor]) < squarred_t:
            idx[i] = 0
        else:
            cursor = i

    return idx


def load_data(folder):
    logger.info('Loading file names')

    p = path.Path(folder)
    data = list()
    data.append(list())
    data.append(list())
    for file in p.iterdir():
        if file.is_file() and '.jpg' in file.name:
            c = int(re.search('(?<=c)\d', file.name).group(0))
            data[c].append({'file': 'imgs/' + file.name,
                            'time': float(re.search('\d*(?=us)', file.name).group(0))*1e-6})

    return data


def find_closest_imgs(timestamps, data, threshold=1):

    t_cam_0 = np.array([d['time'] for d in data[0]])
    t_cam_1 = np.array([d['time'] for d in data[1]])

    t_cam_0 = t_cam_0.reshape((t_cam_0.shape[0], 1))
    t_cam_1 = t_cam_1.reshape((t_cam_1.shape[0], 1))
    timestamps = timestamps.reshape((timestamps.shape[0], 1))

    nn_searcher = nn.NearestNeighbors(n_neighbors=1)

    nn_searcher.fit(t_cam_0)
    distance_0, nn_cam_0 = nn_searcher.kneighbors(timestamps, return_distance=True)
    nn_searcher.fit(t_cam_1)
    distance_1, nn_cam_1 = nn_searcher.kneighbors(timestamps, return_distance=True)

    return np.concatenate((nn_cam_0, nn_cam_1), 1), (distance_1+distance_0 < 2*threshold)[:, 0]


def prepar_data(collection_folder, pivot=None, camera_bearing=None, prune_tolerance=1):
    if camera_bearing is None:
        camera_bearing = (np.pi/3, -np.pi/3)

    root_to_folders = os.environ['CMU'] + collection_folder
    coord = pd.read_csv(root_to_folders + 'GPS.txt', sep='\t', usecols=[1, 4, 10, 11, 14])

    timestamps = coord.ix[:, 0].as_matrix()
    quality = coord.ix[:, 1].as_matrix().astype('int')
    north = coord.ix[:, 2].as_matrix().astype('float')
    south = coord.ix[:, 3].as_matrix().astype('float')
    bearing = coord.ix[:, 4].as_matrix().astype('float')

    north = north[quality == 1]
    south = south[quality == 1]
    bearing = bearing[quality == 1]
    timestamps = timestamps[quality == 1]

    idx = prune_path(np.concatenate((north.reshape((north.shape[0], 1)), south.reshape((north.shape[0], 1))), 1),
                     tolerance=prune_tolerance)

    north = north[idx == 1]
    south = south[idx == 1]
    bearing = bearing[idx == 1]
    timestamps = timestamps[idx == 1]

    if pivot is None:
        pivot = (589292.967387208, 4477474.976132023) # data_collection_20100915: (589292.967387208, 4477474.976132023)

    north -= pivot[0]
    south -= pivot[1]

    data = load_data(root_to_folders + 'imgs')
    nn, time_outliers = find_closest_imgs(timestamps, data)

    nn = nn[time_outliers, :]
    bearing = bearing[time_outliers]
    south = south[time_outliers]
    north = north[time_outliers]

    files_name = list()
    coord_file = list()
    for i, inn in enumerate(nn):
        files_name.append(data[0][inn[0]]['file'])
        files_name.append(data[1][inn[1]]['file'])
        coord_file.append( (north[i], south[i], in_range_2pi(bearing[i] + camera_bearing[0])) )
        coord_file.append( (north[i], south[i], in_range_2pi(bearing[i] + camera_bearing[1])) )

    return files_name, coord_file


def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    buffer = tuple()
    for name, mod in sample_batched.items():
        if name not in ('coord'):
            '''
            min_v = mod.min()
            mod -= min_v
            max_v = mod.max()
            mod /= max_v
            '''
            buffer += (mod,)

    images_batch = torch.cat(buffer, 0)
    grid = torchvis.utils.make_grid(images_batch, nrow=4)

    plt.imshow(grid.numpy().transpose((1, 2, 0)))

def saveas_txt(data, file):
    with open(file, 'w') as f:
        for d in data:
            if isinstance(d, (list, tuple)):
                line = ''
                for sd in d:
                    line += '{},'.format(sd)
                line = line[:-1] + '\n'

            else:
                line = '{}\n'.format(d)
            f.write(line)

if __name__ == '__main__':
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    os.environ['CMU'] = "/mnt/anakim/data/"

    files_name, coord_file = prepar_data('data_collection_20101221/', prune_tolerance=30)
    print("Saving files")
    saveas_txt(files_name, 'test.txt')
    saveas_txt(coord_file, 'test_coordxImbearing.txt')

    """
    root_to_folders = os.environ['CMU'] + 'data_collection_20100915/'
    coord_file = 'GPS.txt'
    coord = pd.read_csv(root_to_folders + coord_file, sep='\t', usecols=[1, 4, 10, 11, 14])

    timestamp = coord.ix[:, 0].as_matrix()
    quality = coord.ix[:, 1].as_matrix().astype('int')
    north = coord.ix[:, 2].as_matrix().astype('float')
    south = coord.ix[:, 3].as_matrix().astype('float')
    bearing = coord.ix[:, 4].as_matrix().astype('float')

    south = south[quality == 1]
    north = north[quality == 1]
    bearing = bearing[quality == 1]

    pivot_south = south[0]
    pivot_north = north[0]
    south -= pivot_south
    north -= pivot_north

    ax.scatter(south, north, bearing, depthshade=True, s=100, marker='.')

    root_to_folders = os.environ['CMU'] + 'data_collection_20101221/'
    coord_file = 'GPS.txt'
    coord = pd.read_csv(root_to_folders + coord_file, sep='\t', usecols=[4, 10, 11, 14], dtype=np.float64)

    quality = coord.ix[:, 0].as_matrix().astype('int')
    north = coord.ix[:, 1].as_matrix().astype('float')
    south = coord.ix[:, 2].as_matrix().astype('float')
    bearing = coord.ix[:, 3].as_matrix().astype('float')

    south = south[quality == 1]
    north = north[quality == 1]
    bearing = bearing[quality == 1]

    south -= pivot_south
    north -= pivot_north

    ax.scatter(south, north, bearing, color='r',depthshade=True, s=100, marker='*')

    plt.show()
    """
