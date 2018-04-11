import setlog
import yaml
import torch.utils.data
import copy
import matplotlib.pyplot as plt
import networks.Descriptor              # Needed for class creation with eval
import trainer.TripletTrainers          # Needed for class creation with eval
import score.Functions                  # Needed for class creation with eval
import datasets.Robotcar                # Needed for class creation with eval


logger = setlog.get_logger(__name__)


class Base:
    def __init__(self, **kwargs):
        self.root = kwargs.pop('root', None)
        self.param_file = kwargs.pop('dataset_file', 'params.yaml')
        self.trainer_file = kwargs.pop('trainer_file', 'trainer.yaml')
        self.cnn_file = kwargs.pop('cnn_type', 'cnn.yaml')
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        with open(self.root + self.cnn_file, 'rt') as f:
            network_params = yaml.safe_load(f)
            logger.debug('cnn param files {} is:'.format(self.root + self.cnn_file))
            logger.debug(yaml.safe_dump(network_params))
        self.network = eval(network_params['class'])(**network_params['param_class'])

        with open(self.root + self.trainer_file, 'rt') as f:
            trainer_params = yaml.safe_load(f)
            logger.debug('trainer param files is {}:'.format(self.root + self.trainer_file))
            logger.debug(yaml.safe_dump(trainer_params))
        self.trainer = eval(trainer_params['class'])(network=self.network, **trainer_params['param_class'])
        self.eval_func = eval(trainer_params['eval_class'])(**trainer_params['param_eval_class'])
        self.test_func = dict()
        for test_func_name in trainer_params['test_func']:
            self.test_func[test_func_name] = eval(trainer_params['test_func'][test_func_name]['class'])\
                (
                    **trainer_params['test_func'][test_func_name]['param_class']
                )

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
        self.results = params.pop('score', None)
        self.stop_criteria_epsilon = params.pop('stop_criteria_epsilon', 1e-6)
        self.min_value_to_stop = params.pop('min_value_to_stop', 10)
        self.sucess_bad_epoch = params.pop('sucess_bad_epoch', 2)
        params.pop('saved_files')
        if params:
            logger.error('Unexpected **params: %r' % params)
            raise TypeError('Unexpected **params: %r' % params)

        if self.curr_epoch != 0:
            self.load()

    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def save(self, datas):
        self.params['saved_files'] = dict()
        for name, data in datas.items():
            self.params['saved_files'][name] = self.root + name + '.pth'
            torch.save(data, self.root + name + '.pth')

        self.params['curr_epoch'] = self.curr_epoch
        self.params['score'] = self.results
        with open(self.root + self.param_file, 'wt') as f:
            f.write(yaml.safe_dump(self.params))
        logger.info('Checkpoint saved at epoch {}'.format(self.curr_epoch))

    def load(self):
        datas = dict()
        for name, file in self.params['saved_files'].items():
            datas[name] = torch.load(file)
        self.trainer.load(datas)

    def plot(self, **kwargs):
        print_loss = kwargs.pop('print_loss', True)
        print_val = kwargs.pop('print_val', True)
        print_score = kwargs.pop('print_score', True)
        size_dataset = kwargs.pop('size_dataset', 0)

        if print_loss:
            nbtch = round(size_dataset/self.batch_size)
            print(nbtch, size_dataset, self.batch_size)
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
            print('Validation score is {}'.format(self.trainer.best_net[0]))
            scores_to_plot = list()
            for i, (name, vals) in enumerate(self.results.items()):
                try:
                    len(vals)
                except TypeError:
                    print(self.test_func[name], '= {}'.format(vals))
                else:
                    scores_to_plot.append((name, vals))
            if len(scores_to_plot):
                f, axes = plt.subplots(len(scores_to_plot) + 1, sharex=True)
                for i, (name, vals) in enumerate(scores_to_plot):
                    print(vals)
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
        seuil = max(int(0.1*len(derive)), min(len(derive), self.min_value_to_stop))
        if seuil < self.min_value_to_stop:
            logger.info('[STOP CRITERIA] Not enought values to compute stop criteria ({} values)'.format(seuil))
            return False  # Continue training

        moy = sum(derive[:seuil])/seuil
        logger.info('[STOP CRITERIA] Mean of latest derivative value is {}'.format(moy))
        if criteria == 'minimize':
            if moy + self.stop_criteria_epsilon > 0:
                return True  # Stop training
            else:
                return False  # Continue training
        elif criteria == 'maximize':
            if moy - self.stop_criteria_epsilon > 0:
                return False  # Continue training
            else:
                return True  # Stop training


if __name__ == '__main__':
    pass
