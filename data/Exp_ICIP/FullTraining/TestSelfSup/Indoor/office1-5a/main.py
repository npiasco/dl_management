#!/usr/bin/env python
import os, sys
import setlog


conf_file = os.environ['DEV'] + 'dl_management/.log/logging.yaml'
save_file = os.path.abspath(sys.argv[0])[:-len(sys.argv[0])] + 'log/'
setlog.reconfigure(conf_file, save_file)


import system.PoseRegression as System


if __name__ == '__main__':
    scene = 'indoor/office1-5a'
    machine = System.MultNet(root=os.path.abspath(sys.argv[0])[:-len(sys.argv[0])],
                             #trainer_file= '../relative_pnp.yaml',
                             #trainer_file='../self_multiscale_depth_trainer.yaml',
                             #trainer_file='../nn_index.yaml',
                             trainer_file='../pnp-5-images.yaml',
                             #trainer_file='../pnp-1-image.yaml',
                             dataset_file = '../../../../datasets/' + scene + '.yaml',
                             cnn_type='../multiscale_cnn.yaml'
                             )
    action = input('Exec:\n[t]\ttrain\n[e]\ttest\n[p]\tprint (console)\n[P]\tprint (full)\n[ ]\ttrain+test\n')
    if action == 't':
        machine.train()
    elif action == 'e':
        machine.test()
        machine.plot(print_loss=False, print_val=False)
    elif action == 'ef':
        machine.test_on_final()
        machine.plot(print_loss=False, print_val=False)
    elif action == 'p':
        machine.plot(print_loss=False, print_val=False)
    elif action == 'P':
        machine.plot()
    elif action == 'm':
        machine.map_print(batch_size=2, aux_mod='rgb')
    elif action == 'mf':
        machine.map_print(final=True, batch_size=2, aux_mod='rgb')
    elif action == 'pose':
        machine.view_localization(pas=3)
    elif action == 'posef':
        machine.view_localization(pas=3, final=True)
    elif action == 'model':
        machine.creat_model()
    elif action == 'modeld':
        machine.creat_model(fake_depth=True)
    elif action == 'modelt':
        machine.creat_model(test=True)
    elif action == 'modeldt':
        machine.creat_model(test=True, fake_depth=True)
    elif action == 'clusters':
        machine.creat_clusters(64, size_feat=256, map_feat='conv7')
    elif action == 'thresh':
        machine.threshold_selection(final=True, dataset='test', load=False, beg=0.0, n_values=2000)
    elif action == 'threshl':
        machine.threshold_selection(final=True, dataset='test', load=True, beg=0.0, n_values=2000)
    elif action == '':
        machine.train()
        machine.test()
        machine.plot(print_loss=False, print_val=False)
    else:
        raise ValueError('Unknown cmd: {}'.format(action))
