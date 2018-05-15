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

        self.alpha = torch.nn.Parameter(torch.FloatTensor(init_weight),
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


def adaptive_triplet_loss(anchor, positives, negatives, margin=0.25, p=2, eps=1e-6):
    d_p = None
    d_n = None
    for positive in positives:
        if d_p is None:
            d_p = func.pairwise_distance(anchor, positive, p, eps)
        else:
            d_p += func.pairwise_distance(anchor, positive, p, eps)
    for negative in negatives:
        if d_n is None:
            d_n = func.pairwise_distance(anchor, negative, p, eps)
        else:
            d_n += func.pairwise_distance(anchor, negative, p, eps)
    l_p = len(positives)
    l_n = len(negatives)
    dist_hinge = torch.clamp(margin + d_p/l_p - d_n/l_n, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss


def triplet_margin_loss(anchor, positives, negatives, margin=0.25, p=2, eps=1e-6, factor=1,swap=False):
    return factor*func.triplet_margin_loss(anchor, positives, negatives, margin=margin, p=p, eps=eps, swap=swap)


def mult_triplet_margin_loss(anchor, positives, negatives, margin=0.25, p=2, eps=1e-6, factor=None, swap=False):
    loss = dict()
    for part, part_factor in factor.items():
        loss[part] = part_factor*func.triplet_margin_loss(anchor[part],
                                                          positives[part],
                                                          negatives[part],
                                                          margin=margin, p=p, eps=eps, swap=swap)
    return sum(loss.values())


def l1_modal_loss(predicted_maps, gt_maps, p=1, factor=3e-4):

    if p == 1:
        loss = factor * func.l1_loss(
            torch.cat(predicted_maps, dim=1),
            torch.cat(gt_maps, dim=1)
        )
    elif p == 2:
        loss = factor * func.mse_loss(
            torch.cat(predicted_maps, dim=1),
            torch.cat(gt_maps, dim=1)
        )
    else:
        raise AttributeError('No behaviour for p = {}'.format(p))

    return loss
