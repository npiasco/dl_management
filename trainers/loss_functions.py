import setlog
import torch
import torch.nn.functional as func


logger = setlog.get_logger(__name__)


def mean_dist(predicted, gt):
    return torch.mean(func.pairwise_distance(
        predicted,
        gt)
    )


def full_pose_loss(predicted, gt, key='full', combine_func=None):

    return combine_func.combine(mean_dist(predicted[key]['p'], gt['p']),
                                mean_dist(predicted[key]['q'], gt['q']))


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


def adaptive_triplet_loss(anchor, positives, negatives, **kwargs):
    margin = kwargs.pop('margin', 0.25)
    p = kwargs.pop('p', 2)
    eps = kwargs.pop('eps', 1e-6)
    swap = kwargs.pop('swap', True)
    adaptive_factor = kwargs.pop('adaptive_factor', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    tt_loss = None
    cpt = 0
    for positive in positives:
        for negative in negatives:
            c_loss = func.triplet_margin_loss(anchor,
                                                   positive,
                                                   negative,
                                                   margin=margin,
                                                   eps=eps,
                                                   p=p,
                                                   swap=swap)
            if tt_loss is None:
                tt_loss = c_loss
            else:
                tt_loss += c_loss

            if adaptive_factor:
                cpt += 1 if c_loss.data[0]>0 else 0
            else:
                cpt += 1

    tt_loss /= cpt if cpt else 1
    return tt_loss


def triplet_margin_loss(anchor, positives, negatives, margin=0.25, p=2, eps=1e-6, factor=1, swap=False):
    return factor*func.triplet_margin_loss(anchor, positives, negatives, margin=margin, p=p, eps=eps, swap=swap)


def mult_triplet_margin_loss(anchor, positives, negatives, margin=0.25, p=2, eps=1e-6, factor=None, swap=False):
    loss = dict()
    for part, part_factor in factor.items():
        loss[part] = part_factor*adaptive_triplet_loss(anchor[part],
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


def diversification_loss(anchor, positives, negatives, **kwargs):
    original_loss = kwargs.pop('original_loss', dict())
    factor = kwargs.pop('factor', 1)
    marge = kwargs.pop('marge', 0.1)

    if isinstance(original_loss['func'], str):
        original_loss['func'] = eval(original_loss['func'])

    main = original_loss['func'](anchor['main'], positives['main'], negatives['main'], **original_loss['param'])
    aux = original_loss['func'](anchor['aux'], positives['aux'], negatives['aux'], **original_loss['param'])
    full = original_loss['func'](anchor['full'], positives['full'], negatives['full'], **original_loss['param'])

    loss = factor*(torch.clamp(full + marge - main, min=0)) # Not shure about that + torch.clamp(full + marge - aux, min=0))
    return loss
