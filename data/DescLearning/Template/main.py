#!/usr/bin/env python
import os, sys
import setlog

file = '.log/logging.yaml'
root = os.environ['DEV'] + 'dl_management/'
setlog.reconfigure(file, root)

import system.DescriptorLearning as System


if __name__ == '__main__':
    machine = System.Default(root=os.path.dirname(sys.argv[0]) + '/')
    action = input('Exec:\n[t]\ttrain\n[e]\ttest\n[p]\tprint (console)\n[P]\tprint (full)\n[ ]\ttrain+test')
    if action == 't':
        machine.train()
    elif action == 'e':
        machine.test()
    elif action == 'p':
        machine.plot(print_loss=False, print_val=False)
    elif action == 'P':
        machine.plot()
    elif action == '':
        machine.train()
        machine.test()
    else:
        raise ValueError('Unknown cmd: {}'.format(action))
