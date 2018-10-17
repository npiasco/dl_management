import setlog
import torch
import torch.nn.functional as func
import torch.autograd as auto
import random as rand


logger = setlog.get_logger(__name__)


def mean_dist(predicted, gt):
    return torch.mean(func.pairwise_distance(
        predicted,
        gt)
    )


def full_pose_loss(predicted, gt, key='full', combine_func=None):

    return combine_func.combine(mean_dist(predicted[key]['p'], gt['p']),
                                mean_dist(predicted[key]['q'], gt['q']))


def minmax_pose_loss(p_ps, p_qs, gt_ps, gt_qs):
    loss = 0
    for i, p_p in enumerate(p_ps):
        p_loss = mean_dist(p_p, gt_ps[i,:])
        q_loss = mean_dist(p_qs[i, :], gt_qs[i, :])
        loss += 0.9*torch.max(torch.stack((p_loss, q_loss), 0)) + 0.1*torch.min(torch.stack((p_loss, q_loss), 0))

    return loss/(i+1)


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


def l1_modal_loss(predicted_maps, gt_maps, **kwargs):
    p = kwargs.pop('p', 1)
    factor = kwargs.pop('factor', 1)
    listed_maps = kwargs.pop('listed_maps', True)
    reg = kwargs.pop('reg', 0)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if listed_maps:
        predicted = torch.cat(predicted_maps, dim=0)
        gt_w_grad = torch.cat(gt_maps, dim=0)
    else:
        predicted = predicted_maps
        gt_w_grad = gt_maps

    gt = gt_w_grad.detach()

    if p == 1:
        loss = factor * func.l1_loss(predicted, gt)
    elif p == 2:
        loss = factor * func.mse_loss(predicted, gt)
    else:
        raise AttributeError('No behaviour for p = {}'.format(p))

    if reg:
        loss += reg * (
            torch.sum(torch.abs(predicted[:, :, :, :-1] - predicted[:, :, :, 1:])) +
            torch.sum(torch.abs(predicted[:, :, :-1, :] - predicted[:, :, 1:, :]))
        ) / predicted.size(0)

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


def GANLoss(*input, **kwargs):
    target_is_real = kwargs.pop('target_is_real', None)
    factor =  kwargs.pop('factor', 1.0)
    mse = kwargs.pop('mse', False)
    multiples_instance = kwargs.pop('multiples_instance', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if multiples_instance:
        f_input = list()
        for seq in input:
            if isinstance(seq, list):
                f_input += seq
            else:
                f_input.append(seq)
        input = torch.cat(f_input, 0)

    if target_is_real:
        target_tensor = torch.rand(input.size())*0.2
    else:
        target_tensor =  torch.rand(input.size())*0.2 + 0.8

    if input.is_cuda:
        target_tensor = target_tensor.cuda()

    target_tensor = auto.Variable(target_tensor)
    if mse:
        loss = func.mse_loss(input, target_tensor) * factor
    else:
        loss = func.binary_cross_entropy(input, target_tensor) * factor
    return loss
