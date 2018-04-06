import setlog


logger = setlog.get_logger(__name__)



class Base:
    def __init__(self, **kwargs):
        self.train_dataset = kwargs.pop('train_dataset', None)
        self.eval_dataset = kwargs.pop('eval_dataset', None)
        self.test_dataset = kwargs.pop('test_dataset', None)
        self.cnn_type = kwargs.pop('cnn_type', None)
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)




if __name__ == '__main__':
    pass
