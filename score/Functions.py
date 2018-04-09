import setlog
import tqdm


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

    @staticmethod
    def rank_score(new_score, old_score):
        return new_score > old_score
