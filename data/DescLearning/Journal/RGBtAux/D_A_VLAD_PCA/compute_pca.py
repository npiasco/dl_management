#!/usr/bin/env python
import os, sys
import setlog


conf_file = os.environ['DEV'] + 'dl_management/.log/logging.yaml'
save_file = os.path.abspath(sys.argv[0])[:-len(sys.argv[0])] + 'log/'
setlog.reconfigure(conf_file, save_file)


import system.DescriptorLearning as System


if __name__ == '__main__':
    machine = System.MultNet(root=os.path.abspath(sys.argv[0])[:-len(sys.argv[0])],
                             cnn_type='cnn.yaml',
                             trainer_file='../trainer_att.yaml',
                             dataset_file='../../../SummerTests/datasets/cmu_lt.yaml')
    machine.compute_PCA(256, desc=['main_out', 'desc'])
    os.system('mv pca_256-D.pth pca_256-D_main.pth')

    machine.compute_PCA(256, desc=['aux_desc'])
    os.system('mv pca_256-D.pth pca_256-D_aux.pth')
