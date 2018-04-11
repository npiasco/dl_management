import setlog
import tqdm
import random


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
        return new_score > old_score


class MeanRecallAtN:
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n', 50)
        self.radius = kwargs.pop('radius', 25)  # In metre

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, ranked_queries):
        logger.info('Computing score')
        scores = [0 for _ in range(self.n)]
        inc_func = lambda i, val, r: val + 1 if i >= r else val
        for query in tqdm.tqdm(ranked_queries):
            for recall in range(1, self.n+1):
                is_ranked = [query[i] < self.radius for i in range(recall)]
                if True in is_ranked:
                    scores = [inc_func(i + 1, val, recall) for i, val in enumerate(scores)]
                    break

        n_query = len(ranked_queries)
        tmp_score = [s/n_query for s in scores]
        self.score = sum(tmp_score)/len(tmp_score)
        return self.score

    def __str__(self):
        return 'Mean Recall @{}'.format(self.n)

    @staticmethod
    def rank_score(new_score, old_score):
        return new_score > old_score


class Recall:
    def __init__(self, **kwargs):
        self.n = kwargs.pop('n', 50)
        self.radius = kwargs.pop('radius', 25)  # In metre

        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

    def __call__(self, ranked_queries):
        logger.info('Computing score')
        scores = [0 for _ in range(self.n)]
        inc_func = lambda i, val, r: val + 1 if i >= r else val
        for query in tqdm.tqdm(ranked_queries):
            for recall in range(1, self.n+1):
                is_ranked = [query[i] < self.radius for i in range(recall)]
                if True in is_ranked:
                    scores = [inc_func(i + 1, val, recall) for i, val in enumerate(scores)]
                    break
        n_query = len(ranked_queries)
        self.score = [s / n_query for s in scores]
        return self.score

    def __str__(self):
        return 'Recall (computed up to {})'.format(self.n)


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