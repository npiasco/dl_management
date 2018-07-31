#!/usr/bin/env python
import os, sys
import setlog


conf_file = os.environ['DEV'] + 'dl_management/.log/logging.yaml'
save_file = os.path.abspath(sys.argv[0])[:-len(sys.argv[0])] + 'log/'
setlog.reconfigure(conf_file, save_file)


import system.DescriptorLearning as System


if __name__ == '__main__':
    machine = System.Default(root=os.path.abspath(sys.argv[0])[:-len(sys.argv[0])],
                             dataset_file='../../../../datasets/default.yaml',
                             trainer_file='../trainer.yaml')
    action = input('Exec:\n[t]\ttrain\n[e]\ttest\n[p]\tprint (console)\n[P]\tprint (full)\n[ ]\ttrain+test\n')
    if action == 't':
        machine.train()
    elif action == 'e':
        machine.test()
        machine.plot(print_loss=False, print_val=False)
    elif action == 'p':
        machine.plot(print_loss=False, print_val=False)
    elif action == 'P':
        machine.plot()
    elif action == '':
        machine.train()
        machine.test()
        machine.plot(print_loss=False, print_val=False)
    elif action == 's':
        machine.serialize_net(final=False)
    elif action == 'sf':
        machine.serialize_net(final=True)
    elif action == 'm':
        machine.map_print('Main', final=False)
    elif action == 'mf':
        machine.map_print('Main', final=True)
    elif action == 'jet':
        machine.map_print()
    elif action == 'dataset':
        machine.print('train')
    elif action == 'testq':
        machine.print('test_query')
    elif action == 'testd':
        machine.print('test_data')
    else:
        raise ValueError('Unknown cmd: {}'.format(action))
