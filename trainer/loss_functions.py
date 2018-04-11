import setlog
import torch
import torch.nn.functional as func


logger = setlog.get_logger(__name__)


def mean_dist(predicted, gt):
    return torch.mean(func.pairwise_distance(
        predicted,
        gt)
    )


class AlphaWeights:
    def __init__(self, init_weight=(0, -3), cuda=True):
        self.alpha = torch.nn.Parameter(torch.Tensor(init_weight),
                                        requires_grad=True)
        self.cuda = cuda

    def cuda_func(self, elem):
        return elem.cuda() if self.cuda else elem

    def combine(self, l1, l2):
        return l1 * torch.exp(-1 * self.cuda_func(self.alpha[0])) + \
               l2 * torch.exp(-1 * self.cuda_func(self.alpha[1]))

    @property
    def params(self):
        return [
            {'params': self.cuda_func(self.alpha)}
        ]


class BetaWeights:
    def __init__(self, init_weight=312, cuda=True):
        self.beta = init_weight
        self.cuda = cuda

    def combine(self, l1, l2):
        return l1 + l2 * self.beta

    @property
    def params(self):
        return []
