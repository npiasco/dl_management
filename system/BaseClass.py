import setlog
import yaml
import torch.utils.data
import copy
import matplotlib.pyplot as plt
import torch.utils as utils
import tqdm
import sys
import os
import torch.optim.lr_scheduler as lr_scheduler
import networks.Descriptor              # Needed for class creation with eval
import networks.Pose                    # Needed for class creation with eval
import trainers.TripletTrainers          # Needed for class creation with eval
import trainers.PoseTrainers             # Needed for class creation with eval
import score.Functions                  # Needed for class creation with eval
import datasets.Robotcar                # Needed for class creation with eval
import datasets.SevenScene              # Needed for class creation with eval


logger = setlog.get_logger(__name__)


class Base:
    def __init__(self, **kwargs):
        self.root = kwargs.pop('root', None)
        self.param_file = kwargs.pop('param_file', 'params.yaml')
        self.trainer_file = kwargs.pop('trainer_file', 'trainer.yaml')
        self.cnn_file = kwargs.pop('cnn_type', 'cnn.yaml')
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        with open(self.root + self.cnn_file, 'rt') as f:
            self.network_params = yaml.safe_load(f)
            logger.debug('cnn param files {} is:'.format(self.root + self.cnn_file))
            logger.debug(yaml.safe_dump(self.network_params))

        with open(self.root + self.trainer_file, 'rt') as f:
            self.trainer_params = yaml.safe_load(f)
            logger.debug('trainer param files is {}:'.format(self.root + self.trainer_file))
            logger.debug(yaml.safe_dump(self.trainer_params))

        self.eval_func = eval(self.trainer_params['eval_class'])(**self.trainer_params['param_eval_class'])
        self.test_func = dict()
        for test_func_name, fonction in self.trainer_params['test_func'].items():
            self.test_func[test_func_name] = eval(fonction['class'])(**fonction['param_class'])

        with open(self.root + self.param_file, 'rt') as f:
            params = yaml.safe_load(f)
            logger.debug('system param files is {}:'.format(self.root + self.param_file))
            logger.debug(yaml.safe_dump(params))
        self.params = copy.deepcopy(params)
        self.curr_epoch = params.pop('curr_epoch', 0)
        self.max_epoch = params.pop('max_epoch', 1000)
        self.batch_size = params.pop('batch_size', 64)
        self.num_workers = params.pop('num_workers', 8)
        self.shuffle = params.pop('shuffle', True)
        self.stop_criteria_epsilon = params.pop('stop_criteria_epsilon', 1e-6)
        self.min_value_to_stop = params.pop('min_value_to_stop', 10)
        self.sucess_bad_epoch = params.pop('sucess_bad_epoch', 2)
        self.score_file = params.pop('score_file', None)
        self.lr_scheduler = params.pop('lr_scheduler', None)
        self.training_mod = params.pop('training_mod', list())
        self.testing_mod = params.pop('testing_mod', list())

        params.pop('saved_files', None)
        if params:
            logger.error('Unexpected **params: %r' % params)
            raise TypeError('Unexpected **params: %r' % params)

        self.results = None
        self.data = None
        self.trainer = None

    @staticmethod
    def creat_dataset(params, env_var):
        transform = dict()
        for name, content in params['param_class']['transform'].items():
            transform[name] = list()
            for diff_tf in content:
                if diff_tf['param_class']:
                    transform[name].append(eval(diff_tf['class'])(**diff_tf['param_class']))
                else:
                    transform[name].append(eval(diff_tf['class'])())
        params['param_class'].pop('transform')
        name = env_var + params['param_class']['root']
        params['param_class'].pop('root')
        return eval(params['class'])(root=name, transform=transform, **params['param_class'])

    def train(self):
        logger.info('Initial saving before training.')
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

            if self.lr_scheduler is not None:
                last_epoch = self.curr_epoch - 1 # self.curr_epoch if self.curr_epoch != 0 else -1
                if hasattr(self.trainer, 'optimizers'):
                    schedulers = dict()
                    for name, t_scheduler in self.lr_scheduler.items():
                        schedulers[name] = eval(t_scheduler['class'])(self.trainer.optimizers[name],
                                                                      **t_scheduler['param'],
                                                                      last_epoch=last_epoch)

                else:
                    scheduler = eval(self.lr_scheduler['class'])(self.trainer.optimizer,
                                                                 **self.lr_scheduler['param'],
                                                                 last_epoch=last_epoch)


            for ep in range(self.curr_epoch, self.max_epoch):
                logger.info('Training network for ep {}'.format(ep + 1))

                if self.lr_scheduler is not None:
                    if hasattr(self.trainer, 'optimizers'):
                        for scheduler in schedulers.values():
                            scheduler.step()
                            logger.info('Learning rate are {}'.format(scheduler.get_lr()))
                    else:
                        scheduler.step()
                        logger.info('Learning rate are {}'.format(scheduler.get_lr()))

                for b in tqdm.tqdm(dtload):
                    self.trainer.train(b)
                self.curr_epoch += 1
                self.trainer.eval(dataset=self.data['val'],
                                  score_function=self.eval_func,
                                  ep=self.curr_epoch)

                loss = [sum(elem) for elem in zip(*self.trainer.loss_log.values())]
                criteria_loss.pop(0)
                criteria_loss.append(self.compute_stop_criteria(loss, float.__lt__))
                criteria_val.pop(0)
                criteria_val.append(self.compute_stop_criteria(self.trainer.val_score, self.eval_func.rank_score))
                if False not in criteria_loss + criteria_val:
                    break
        except KeyboardInterrupt:
            logger.error('Aborting training with interruption:\n{}'.format(sys.exc_info()[0]))
        finally:
            self.save(self.trainer.serialize())

    def test(self):
        self.results = self.trainer.test(dataset=self.data['test'],
                                         score_functions=self.test_func)
        self.save(self.trainer.serialize())
        return self.results

    def save(self, datas):
        self.params['saved_files'] = dict()
        for name, data in datas.items():
            self.params['saved_files'][name] = name + '.pth'
            torch.save(data, self.root + name + '.pth.tmp')
            os.system('mv {} {}'.format(self.root + name + '.pth.tmp', self.root + name + '.pth'))
            logger.info('Saved {}'.format(self.root + name + '.pth'))

        self.params['curr_epoch'] = self.curr_epoch
        self.params['score_file'] = 'score_file.pth'
        torch.save(self.results, self.root + 'score_file.pth')
        with open(self.root + self.param_file, 'wt') as f:
            f.write(yaml.safe_dump(self.params))
        logger.info('Checkpoint saved at epoch {}'.format(self.curr_epoch))

    def serialize_net(self, final=False):
        tmp_net = copy.deepcopy(self.trainer.network)
        if not final:
            tmp_net.load_state_dict(self.trainer.best_net[-1])
        serlz = tmp_net.cpu().full_save()
        for name, data in serlz.items():
            torch.save(data, self.root + name + '.pth')

    def load(self):
        datas = dict()
        for name, file in self.params['saved_files'].items():
            logger.info('Loading ' + name)
            datas[name] = torch.load(self.root + file)
        self.trainer.load(datas)
        self.results = torch.load(self.root + self.params['score_file'])

    def plot(self, **kwargs):
        print_loss = kwargs.pop('print_loss', True)
        print_val = kwargs.pop('print_val', True)
        print_score = kwargs.pop('print_score', True)
        size_dataset = kwargs.pop('size_dataset', 0)

        if print_loss:
            nbtch = round(size_dataset/self.batch_size)
            losses = self.trainer.loss_log
            f, axes = plt.subplots(len(losses) + 1, sharex=True)
            for i, (name, vals) in enumerate(losses.items()):
                axes[i].plot(vals)
                axes[i].set_title(name)

            handles = list()
            for name, vals in losses.items():
                handle, = axes[-1].plot(vals, label=name)
                handles.append(handle)

            legend = plt.legend(handles=handles)
            axes[-1].add_artist(legend)

            f, axes = plt.subplots(len(losses) + 1, sharex=True)
            for i, (name, vals) in enumerate(losses.items()):
                axes[i].plot(
                    [sum(vals[i * nbtch:(i + 1) * nbtch]) / nbtch for i in range(len(vals) // nbtch)])
                axes[i].set_title(name)

            handles.clear()
            for name, vals in losses.items():
                handle, = axes[-1].plot(
                    [sum(vals[i * nbtch:(i + 1) * nbtch]) / nbtch for i in range(len(vals) // nbtch)],
                    label=name)
                handles.append(handle)

            legend = plt.legend(handles=handles)
            axes[-1].add_artist(legend)

        if print_val:
            plt.figure()
            plt.plot(self.trainer.val_score)
            plt.title('Validation score')

        if print_score:
            logger.info('Validation score is {}'.format(self.trainer.best_net[0]))

            try:
                logger.info('At epoch {}'.format(
                    len(self.trainer.val_score) - 1 - self.trainer.val_score[::-1].index(self.trainer.best_net[0])
                ))
            except ValueError:
                logger.info('Initial weights')

            scores_to_plot = list()
            for i, (name, vals) in enumerate(self.results.items()):
                try:
                    len(vals)
                except TypeError:
                    logger.info('[{}] {} = {}'.format(self.dataset_file, self.test_func[name], vals))

                else:
                    scores_to_plot.append((name, vals))
            if len(scores_to_plot):
                f, axes = plt.subplots(len(scores_to_plot) + 1, sharex=True)
                for i, (name, vals) in enumerate(scores_to_plot):
                    axes[i].plot(vals)
                    axes[i].set_title(name)
                handles = list()
                for name, vals in scores_to_plot:
                    handle, = axes[-1].plot(vals, label=name)
                    handles.append(handle)

                legend = plt.legend(handles=handles)
                axes[-1].add_artist(legend)

        plt.show()

    def compute_stop_criteria(self, seq, criteria):
        derive = [seq[i] - seq[i-1] for i in reversed(range(len(seq)-1))]
        trend = float(sum(1 for v in derive if v > 0) - sum(1 for v in derive if v < 0))

        seuil = max(int(0.1*len(derive)), min(len(derive), self.min_value_to_stop))
        if seuil < self.min_value_to_stop:
            logger.info('[STOP CRITERIA] Not enought values to compute stop criteria ({} values)'.format(seuil))
            return False  # Continue training

        moy = sum(derive[:seuil])/seuil
        logger.info('[STOP CRITERIA] Mean of latest derivative value is {}. Trend is {}'.format(moy, trend))
        if abs(moy) < self.stop_criteria_epsilon:
            logger.warning('Derivative values vanishing')
            return True  # Stop training, derivative vanishing

        if criteria(moy, 0): # or criteria(trend, 0):
            return False  # Continue training
        else:
            logger.warning('Continuation criteria violated')
            return True  # Stop training


if __name__ == '__main__':
    pass
