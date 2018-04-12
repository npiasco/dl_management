import setlog
import yaml
import os
import system.BaseClass as BaseClass
import datasets.SevenScene              # Needed for class creation with eval


logger = setlog.get_logger(__name__)


class Default(BaseClass.Base):
    def __init__(self, **kwargs):
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        BaseClass.Base.__init__(self, **kwargs)

        env_var = os.environ['SEVENSCENES']

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)
            logger.debug('dataset param files {} is:'.format(self.root + self.dataset_file))
            logger.debug(yaml.safe_dump(dataset_params))

        self.data = dict()
        self.data['train'] = self.creat_dataset(dataset_params['train'], env_var)
        self.data['test'] = self.creat_dataset(dataset_params['test'], env_var)
        self.data['val'] = self.creat_dataset(dataset_params['val'], env_var)

        self.training_mod = dataset_params['training_mod']
        self.testing_mod = dataset_params['testing_mod']

    def train(self):
        self.data['train'].used_mod = self.training_mod
        self.data['val'].used_mod = self.testing_mod
        BaseClass.Base.train(self)

    def test(self):
        self.data['test'].used_mod = self.testing_mod
        BaseClass.Base.test(self)

    def plot(self, **kwargs):
        BaseClass.Base.plot(self, **kwargs, size_dataset=len(self.data['train']))


if __name__ == '__main__':
    system = Default(root=os.environ['DATA'] + 'testing_pose/')
    system.train()
    system.test()
    system.plot()
