#!/usr/bin/env python
import os, sys
import setlog
import yaml
import matplotlib.pyplot as plt
import csv
import torch
import argparse
import collections as coll

conf_file = os.environ['DEV'] + 'dl_management/.log/logging.yaml'
save_file = os.path.abspath(sys.argv[0])[:-len(sys.argv[0])] + 'log/'
setlog.reconfigure(conf_file, save_file)


import system.DescriptorLearning as System
import score.Functions


parser = argparse.ArgumentParser(description="Print results")
parser.add_argument("input_file", metavar="Input file", help="Input yaml file descriptor")

args = parser.parse_args()

params_file = args.input_file

with open(params_file, 'rt') as f:
    params = yaml.safe_load(f)

dataset = params['dataset']
dataset_folder_name = 'Results_{}'.format(dataset.replace('.yaml', ''))
ranking = dict()
for net_name, net in params['nets'].items():
    ranking[net_name] = list()

    with open(net['root'] + 'res_file.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for elem in spamreader:
            ranking[net_name].append(elem)

machine = eval(net['class'])(root=net['root'],
                             dataset_file= '../'*net['root'].replace('../', '').count('/')  + '../SummerTests/datasets/{}_aff.yaml'.format(dataset),
                             **net['param_class'])
top_n = 1
thresh_dist = 25
test_dataset = machine.data['test']
try:
    os.mkdir(dataset_folder_name + '/ranking')
except FileExistsError:
    print('Dir already created')

ordered_name = coll.OrderedDict()
for net_name in ranking.keys():
    ordered_name[net_name] = 0

for i, im_request in enumerate(test_dataset['queries']):
    try:
        os.mkdir(dataset_folder_name + '/ranking/q{}'.format(i))
    except FileExistsError:
        print('Dir already created')

    name_im = dataset_folder_name + '/ranking/q{}'.format(i) + '/request.jpg'
    im_request['rgb'].save(name_im)

    for net_name, rank in ranking.items():
        for j in range(top_n):
            n_data_im, dist = ranking[net_name][i][j].replace('(', '').replace(')', '').split(',')
            n_data_im = int(n_data_im)
            dist = float(dist)
            print(dist)
            good_match = int(dist < thresh_dist)
            if j == 0:
                ordered_name[net_name] = good_match

            name_im = dataset_folder_name + '/ranking/q{}'.format(i) + '/{}_{}_{}.jpg'.format(j, net_name, 'MATCH' if good_match else 'NO')
            test_dataset['data'][n_data_im]['rgb'].save(name_im)
    signature = ''
    for val in ordered_name.values():
        signature += str(val)

    os.rename(dataset_folder_name + '/ranking/q{}'.format(i), dataset_folder_name + '/ranking/q{}_{}'.format(i, signature))
