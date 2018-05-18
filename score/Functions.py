import setlog
import tqdm
import random
import numpy as np


logger = setlog.get_logger(__name__)


class RecallAtN:
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n', 5)
        self.radius = kwargs.pop('radius', 25)  # In metre

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, ranked_queries):
        logger.info('Computing score')
        score = 0
        for query in tqdm.tqdm(ranked_queries):
            is_ranked = [query[i] < self.radius for i in range(self.n)]
            if True in is_ranked:
                score += 1

        self.score = score/len(ranked_queries)
        return self.score

    def __str__(self):
        return 'Recall @{}'.format(self.n)

    @staticmethod
    def rank_score(new_score, old_score):
        if old_score is None:
            return True
        else:
            return new_score >= old_score


class MeanRecallAtN:
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n', 50)
        self.radius = kwargs.pop('radius', 25)  # In metre

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, ranked_queries):
        logger.info('Computing score')
        scores = [0 for _ in range(self.n)]
        for query in tqdm.tqdm(ranked_queries):
            for recall in range(1, self.n+1):
                is_ranked = [query[i] < self.radius for i in range(recall)]
                if True in is_ranked:
                    scores = [self.inc_func(i + 1, val, recall) for i, val in enumerate(scores)]
                    break

        n_query = len(ranked_queries)
        tmp_score = [s/n_query for s in scores]
        self.score = sum(tmp_score)/len(tmp_score)
        return self.score

    def __str__(self):
        return 'Mean Recall @{}'.format(self.n)

    @staticmethod
    def inc_func(i, val, r):
        if i >= r:
            return val + 1
        else:
            return val

    @staticmethod
    def rank_score(new_score, old_score):
        if old_score is None:
            return True
        else:
            return new_score >= old_score


class Recall:
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n', 50)
        self.radius = kwargs.pop('radius', 25)  # In metre

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, ranked_queries):
        logger.info('Computing score')
        scores = [0 for _ in range(self.n)]
        for query in tqdm.tqdm(ranked_queries):
            for recall in range(1, self.n+1):
                is_ranked = [query[i] < self.radius for i in range(recall)]
                if True in is_ranked:
                    scores = [self.inc_func(i + 1, val, recall) for i, val in enumerate(scores)]
                    break
        n_query = len(ranked_queries)
        self.score = [s / n_query for s in scores]
        return self.score

    def __str__(self):
        return 'Recall (computed up to {})'.format(self.n)

    @staticmethod
    def inc_func(i, val, r):
        if i >= r:
            return val + 1
        else:
            return val


class GlobalPoseError:
    def __init__(self, **kwargs):
        self.pooling_type = kwargs.pop('pooling_type', 'median')
        self.data_type = kwargs.pop('data_type', 'position')
        if kwargs:
            logger.error('Unexpected **kwargs: %r' % kwargs)
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, errors):
        if self.pooling_type == 'median':
            return np.median(errors[self.data_type])
        elif self.pooling_type == 'mean':
            return sum(errors[self.data_type])/len(errors[self.data_type])
        else:
            logger.error('Unknown pooling named {}'.format(self.pooling_type))
            raise ValueError('Unknown pooling named {}'.format(self.pooling_type))

    def __str__(self):
        return '{} error on {}'.format(self.pooling_type.capitalize(), self.data_type)

    @staticmethod
    def rank_score(new_score, old_score):
        if old_score is None:
            return True
        else:
            return new_score <= old_score


if __name__ == '__main__':
    n_queries = 200
    n_data = 1300
    ranked_q = [
        [random.random() for _ in range(n_data)]
        for _ in range(n_queries)
    ]
    print(ranked_q)
    func = Recall(n=50)
    print(func(ranked_q))
