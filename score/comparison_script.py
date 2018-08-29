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


parser = argparse.ArgumentParser(description="Compute networks results on a given dataset")
parser.add_argument("input_file", metavar="Input file", help="Input yaml file descriptor")
parser.add_argument("--compute", dest='compute', action='store_true', help="Recompute the score for all the networks")
parser.add_argument("--no-compute", dest='compute', action='store_false', help="Do not recompute the score")
parser.set_defaults(compute=True)
parser.add_argument("--plot", dest='plot', action='store_true', help="Plot plotable results")
parser.add_argument("--no-plot", dest='plot', action='store_false', help="No plot")
parser.set_defaults(plot=False)

args = parser.parse_args()

params_file = args.input_file

with open(params_file, 'rt') as f:
    params = yaml.safe_load(f)

folder_name = 'Results_{}'.format(params['dataset'].replace('.yaml','').replace('datasets','').replace('/',''))
try:
    os.mkdir(folder_name)
except FileExistsError:
    print('Dir already created')

result_file = open(folder_name +'/raw_result.txt'.format(params['dataset']), 'w')

test_func = dict()
for test_func_name, fonction in params['test_func'].items():
    test_func[test_func_name] = eval(fonction['class'])(**fonction['param_class'])

if args.compute:
    results = dict()
    for net_name, net in params['nets'].items():
        machine = eval(net['class'])(root=net['root'],
                                     dataset_file= '../'*net['root'].count('/')  + params['dataset'],
                                     **net['param_class'])
        machine.test_func = test_func
        results[net_name] = machine.test()

    torch.save(results, folder_name + '/results.pth')

results = torch.load(folder_name + '/results.pth')

scores_to_plot = dict()
for metric_name in test_func:
    f_time = True
    for network_name, network_result in results.items():
        try:
            len(network_result[metric_name])
        except TypeError:
            if f_time:
                f_time = False
                pstr = str(test_func[metric_name])
                if args.plot:
                    print(pstr)
                result_file.write(pstr + '\n')
            pstr = '[{}] = {}'.format(network_name, network_result[metric_name])
            if args.plot:
                print(pstr)
            result_file.write(pstr + '\n')
        else:
            if len(scores_to_plot.setdefault(metric_name, list())) == 0:
                try:
                    os.mkdir(folder_name + '/{}'.format(metric_name))
                except FileExistsError:
                    print('Dir already created')

            scores_to_plot[metric_name].append({'name': network_name, 'val': network_result[metric_name]})
            with open(folder_name + '/{}/{}.csv'.format(metric_name, network_name), 'w') as new_f:
                spamwriter = csv.writer(new_f, delimiter=',')
                spamwriter.writerow(network_result[metric_name])

result_file.close()

if args.plot:
    if len(scores_to_plot):
        for metric_name, score_values in scores_to_plot.items():
            plt.figure()
            for networks in score_values:
                plt.plot(networks['val'], label=networks['name'])
            plt.legend()

    plt.show()
