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
    def __init__(self, init_weight=(0, -3), cuda=False):

        self.alpha = torch.nn.Parameter(torch.Tensor(init_weight),
                                        requires_grad=True)
        if cuda:
            self.cuda()

    def combine(self, l1, l2):
        return l1 * torch.exp(-1 * self.alpha[0]) + \
               l2 * torch.exp(-1 * self.alpha[1])

    @property
    def params(self):
        return [
            {'params': self.alpha}
        ]

    def cuda(self):
        self.alpha.data = self.alpha.data.cuda()

    def cpu(self):
        self.alpha.data = self.alpha.data.cpu()

    def state_directory(self):
        return self.alpha.data

    def load_state_directory(self, data):
        self.alpha.data = data


class BetaWeights:
    def __init__(self, init_weight=312):
        self.beta = init_weight

    def combine(self, l1, l2, cuda_func):
        return l1 + l2 * self.beta

    @property
    def params(self):
        return []

    def state_directory(self):
        return self.beta

    def load_state_directory(self, data):
        self.beta = data
