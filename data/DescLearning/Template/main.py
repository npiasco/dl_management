#!/usr/bin/env python
import os, sys
import torch
import setlog

conf_file = os.environ['DEV'] + 'dl_management/.log/logging.yaml'
save_file = os.path.abspath(sys.argv[0])[:-len(sys.argv[0])] + 'log/'
setlog.reconfigure(conf_file, save_file)


import system.DescriptorLearning as System


if __name__ == '__main__':
    machine = System.Default(root=os.path.abspath(sys.argv[0])[:-len(sys.argv[0])])
    action = input('''
    Exec:
    [t]\ttrain
    [e]\ttest
    [p]\tprint (console)
    [P]\tprint (full)
    [s]\tserialize net
    [c]\tcreat clusters
    [ ]\ttrain+test
    ''')
    if action == 't':
        machine.train()
    elif action == 'e':
        machine.test()
        machine.plot(print_loss=False, print_val=False)
    elif action == 'p':
        machine.plot(print_loss=False, print_val=False)
    elif action == 'P':
        machine.plot()
    elif action == 's':
        machine.serialize_net()
    elif action == 'test':
        machine.print('test_query')
    elif action == 'train':
        machine.print('train')
    elif action == 'c':
        machine.creat_clusters(size_cluster=64)
    elif action == '':
        machine.train()
        machine.test()
        machine.plot(print_loss=False, print_val=False)
    elif  action == 'S':
        torch.save(machine.network.eval().cpu(), 'default_net.pth')
    elif  action == 'mean':
        machine.compute_mean_std(jobs=8)
        machine.compute_mean_std(jobs=8, training=False, val=False, testing=True)
    elif action == 'dataset':
        machine.print('train')
    else:
        raise ValueError('Unknown cmd: {}'.format(action))
