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
                             trainer_file='../clusters_forward.yaml',
                             dataset_file='../../../SummerTests/datasets/cmu_lt.yaml')


    machine.creat_clusters_auxlad(size_cluster=64, size_feat=256)
    machine.serialize_net(final=True)