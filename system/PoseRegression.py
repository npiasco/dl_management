import setlog
import yaml
import torch.utils as utils
import torch.utils.data
import tqdm
import os
import sys
import system.BaseClass as BaseClass
import datasets.SevenScene              # Needed for class creation with eval


logger = setlog.get_logger(__name__)


class Default(BaseClass.Base):
    def __init__(self, **kwargs):
        self.dataset_file = kwargs.pop('dataset_file', 'dataset.yaml')
        BaseClass.Base.__init__(self, **kwargs)

        with open(self.root + self.dataset_file, 'rt') as f:
            dataset_params = yaml.safe_load(f)
            logger.debug('dataset param files {} is:'.format(self.root + self.dataset_file))
            logger.debug(yaml.safe_dump(dataset_params))

        self.data = dict()
        self.data['train'] = self.creat_dataset(dataset_params['train'])
        self.data['test'] = self.creat_dataset(dataset_params['test'])
        self.data['val'] = self.creat_dataset(dataset_params['val'])

        self.training_mod = dataset_params['training_mod']
        self.testing_mod = dataset_params['testing_mod']

    @staticmethod
    def creat_dataset(params):
        transform = dict()
        for name, content in params['param_class']['transform'].items():
            transform[name] = list()
            for diff_tf in content:
                if diff_tf['param_class']:
                    transform[name].append(eval(diff_tf['class'])(**diff_tf['param_class']))
                else:
                    transform[name].append(eval(diff_tf['class'])())
        params['param_class'].pop('transform')
        name = os.environ['SEVENSCENES'] + params['param_class']['root']
        params['param_class'].pop('root')
        return eval(params['class'])(root=name, transform=transform, **params['param_class'])

    def train(self):
        #TODO: ARMONIZE WITH DESCRIPTOR LERANING
        self.data['train'].used_mod = self.training_mod
        self.data['val'].used_mod = self.testing_mod

        logger.info('Initial saving berfore training.')
        self.save(self.trainer.serialize())

        logger.info('Getting the first validation score...')
        self.trainer.eval(dataset=self.data['val'],
                          score_function=self.eval_func,
                          ep=self.curr_epoch)

        dtload = utils.data.DataLoader(self.data['train'],
                                       batch_size=self.batch_size,
                                       num_workers=self.num_workers,
                                       shuffle=self.shuffle)

        try:
            # End training criteria initiation
            criteria_loss = [False] * self.sucess_bad_epoch
            criteria_val = [False] * self.sucess_bad_epoch

            for ep in range(self.curr_epoch, self.max_epoch):
                logger.info('Training network for ep {}'.format(ep+1))
                for b in tqdm.tqdm(dtload):
                    self.trainer.train(b)
                self.curr_epoch += 1
                self.trainer.eval(self.data['val'],
                                  self.eval_func,
                                  self.curr_epoch)

                loss = [sum(elem) for elem in zip(*self.trainer.loss_log.values())]
                criteria_loss.pop()
                criteria_loss.append(self.compute_stop_criteria(loss, 'minimize'))
                criteria_val.pop()
                criteria_val.append(self.compute_stop_criteria(self.trainer.val_score, 'minimize'))
                if False not in criteria_loss + criteria_val:
                    break
        except:
            logger.error('Aborting training with interruption:\n{}'.format(sys.exc_info()[0]))
        finally:
            self.save(self.trainer.serialize())

    def test(self):
        self.data['test'].used_mod = self.testing_mod
        self.results = self.trainer.test(dataset=self.data['test'],
                                         score_functions=self.test_func)
        self.save(self.trainer.serialize())

    def plot(self, **kwargs):
        BaseClass.Base.plot(self, **kwargs, size_dataset=len(self.data['train']))


if __name__ == '__main__':
    system = Default(root=os.environ['DATA'] + 'testing_pose/')
    #system.train()
    system.test()
    system.plot()
