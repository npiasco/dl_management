import setlog
import yaml
import torch.utils as utils
import torch.utils.data
import tqdm


logger = setlog.get_logger(__name__)


class Base:
    def __init__(self, **kwargs):
        self.root = kwargs.pop('root', None)
        self.param_file = kwargs.pop('dataset_file', 'dataset.yaml')
        self.trainer_file = kwargs.pop('trainer_file', 'trainer.yaml')
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        self.cnn_file = kwargs.pop('cnn_type', 'cnn.yaml')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        network_params = yaml.safe_load(self.cnn_file)
        self.network = eval(network_params['class'])(**network_params['param_class'])
        trainer_params = yaml.safe_load(self.trainer_file)
        self.trainer = eval(trainer_params['class'])(**trainer_params['param_class'])
        self.eval_func = eval(trainer_params['eval_class'])(**trainer_params['param_eval_class'])

        params = yaml.safe_load(self.param_file)
        self.curr_epoch = params.pop('curr_epoch', 0)
        self.max_epoch = params.pop('max_epoch', 1000)
        self.batch_size = params.pop('batch_size', 64)
        self.num_workers = params.pop('num_workers', 8)
        self.shuffle = params.pop('shuffle', True)
        if params:
            raise TypeError('Unexpected **kwargs: %r' % params)

        self.data = dict()

    def train(self):
        dtload = utils.data.DataLoader(self.data['train'],
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=self.shuffle
                                       )
        self.trainer.eval(self.data['test']['query'],
                          self.data['test']['data'],
                          self.eval_func)
        for ep in range(self.curr_epoch, self.max_epoch):
            for b in tqdm.tqdm(dtload):
                self.trainer.train(b)
            self.trainer.eval(self.data['test']['query'],
                              self.data['test']['data'],
                              self.eval_func)

    def test(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def print(self):
        raise NotImplementedError()


if __name__ == '__main__':
    pass
