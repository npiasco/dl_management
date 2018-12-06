#!/usr/bin/env python
import os, sys
import setlog

conf_file = os.environ['DEV'] + 'dl_management/.log/logging.yaml'
save_file = os.path.abspath(sys.argv[0])[:-len(sys.argv[0])] + 'log/'
setlog.reconfigure(conf_file, save_file)


import system.PoseRegression as System


if __name__ == '__main__':
    machine = System.MultNet(root=os.path.abspath(sys.argv[0])[:-len(sys.argv[0])])
    action = input('Exec:\n[t]\ttrain\n[e]\ttest\n[p]\tprint (console)\n[P]\tprint (full)\n[ ]\ttrain+test\n')
    if action == 't':
        machine.train()
    elif action == 'e':
        machine.test()
    elif action == 'p':
        machine.plot(print_loss=False, print_val=False)
    elif action == 'P':
        machine.plot()
    elif action == 'm':
        machine.map_print(batch_size=1)
    elif action == 'mf':
        machine.map_print(final=True, batch_size=4)
    elif action == 'pose':
        machine.view_localization(pas=10)
    elif action == 'posef':
        machine.view_localization(pas=10, final=True)
    elif action == '':
        machine.train()
        machine.test()
    else:
        raise ValueError('Unknown cmd: {}'.format(action))
