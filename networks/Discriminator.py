import setlog
import torch.autograd as auto
import torch.nn as nn
import torch
import networks.Alexnet as Alexnet
import collections as coll
import math


logger = setlog.get_logger(__name__)


class Main(nn.Module):
    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        base_archi = kwargs.pop('base_archi', 'Alexnet')
        base_archi_param = kwargs.pop('base_archi_param',
                                      {
                                          'load_imagenet': False,
                                          'batch_norm': False,
                                          'input_channels': 4,
                                          'end_max_polling': True,
                                          'end_relu': True
                                      })

        self.layers_to_train = kwargs.pop('layers_to_train', 'all')
        input_size = kwargs.pop('input_size', 224)
        self.batch_gan = kwargs.pop('batch_gan', False)

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        if base_archi == 'Alexnet':
            self.feature = Alexnet.Feat(**base_archi_param)
        else:
            raise AttributeError("Unknown base architecture {}".format(base_archi))

        if self.batch_gan:
            input_size = self.batch_gan

        fc_num = round(
            math.floor(input_size/self.feature.down_ratio)**2 * self.feature.final_feat_num()
        )
        logger.info('Discriminator FC layer has {} neurones'.format(fc_num))
        self.classifier = nn.Sequential(
            coll.OrderedDict(
                [
                    ('fc', nn.Linear(fc_num, 1)),
                    ('sig', nn.Sigmoid())
                ]
            )
        )

        logger.info('Classifier architecture:')
        logger.info(self.classifier)

    def forward(self, *x):
        x = torch.cat(x, dim=1)  # Conditional GAN
        if self.batch_gan:
            b, c, w, h = x.size()
            x_class = None
            for i in range(0, w - self.batch_gan, self.batch_gan//2):
                for j in range(0, h - self.batch_gan, self.batch_gan//2):
                    if i + self.batch_gan > w:
                        cropped_x = x[:, :, :-self.batch_gan, :]
                    else:
                        cropped_x = x[:, :, i:i+self.batch_gan, :]
                    if j + self.batch_gan > h:
                        cropped_x = cropped_x[:, :, :, -self.batch_gan]
                    else:
                        cropped_x = cropped_x[:, :, :, j:j+self.batch_gan]

                    x_feat = self.feature(cropped_x)
                    x_feat = x_feat.view(x_feat.size(0), -1)
                    if x_class is None:
                        x_class = self.classifier(x_feat)
                    else:
                        x_class = torch.cat((x_class, self.classifier(x_feat)), dim=1)
            x_class = torch.mean(x_class, dim=1).unsqueeze(1)

        else:
            x_feat = self.feature(x)
            x_feat = x_feat.view(x_feat.size(0), -1)
            x_class = self.classifier(x_feat)

        return x_class

    def get_training_layers(self, layers_to_train=None):
        if not layers_to_train:
            layers_to_train = self.layers_to_train

        if layers_to_train == 'all':
            train_parameters = [{'params': self.classifier.parameters()}] + self.feature.get_training_layers('all')
        else:
            raise KeyError('No behaviour for layers_to_train = {}'.format(layers_to_train))

        return train_parameters

    def full_save(self, discard_tf=False):
        if discard_tf:
            del self.feature.base_archi['jet_tf']
            self.feature.feature = nn.Sequential(self.feature.base_archi)
        return {'feature': self.feature.state_dict(),
                'classifier': self.classifier.state_dict()}


if __name__ == '__main__':
    im_input_size = 224
    tensor_input = torch.rand([10, 1, im_input_size, im_input_size]).cuda()
    tensor_gt = torch.rand([10, 3, im_input_size, im_input_size]).cuda()
    net = Main(input_size=im_input_size, batch_gan=False).cuda()
    feat_output = net(auto.Variable(tensor_input), auto.Variable(tensor_gt))
    print(feat_output)
