#!/usr/bin/env python
import os, sys
import setlog
import yaml
import matplotlib.pyplot as plt
import csv
import torch
import argparse


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

    reader = csv.reader(net['root'] + 'res_file.csv', delimiter=',')

    for elem in reader.next():
        ranking[net_name].append(elem)

machine = eval(net['class'])(root=net['root'],
                             dataset_file='../datasets/{}.yaml'.format(dataset),
                             **net['param_class'])

test_dataset = machine.data['test']