import setlog
import torch
import torch.nn.functional as func
import torch.autograd as auto
import random as rand


logger = setlog.get_logger(__name__)


def SSIM(im1, im2, **kwargs):
    C1 = kwargs.pop('C1', 0.01 ** 2)
    C2 = kwargs.pop('C2', 0.03 ** 2)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    mu_x = func.avg_pool2d(im1, 3, 1)
    mu_y = func.avg_pool2d(im2, 3, 1)

    sigma_x = func.avg_pool2d(im1 ** 2, 3, 1) - mu_x ** 2
    sigma_y = func.avg_pool2d(im2 ** 2, 3, 1) - mu_y ** 2
    sigma_xy = func.avg_pool2d(im1 * im2, 3, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return SSIM_n / SSIM_d


def simple_fact_loss(value, fact=1):

    return torch.mean(value)*fact


def mean_dist(predicted, gt):
    return torch.mean(func.pairwise_distance(
        predicted.view(-1, 1),
        gt.view(-1, 1))
    )


def reproj_on_matching_loss(pc_to_align, pc_ref, T, K, inliers=None, **kwargs):
    '''
    Compute the reprojection that minimise:
    pc_to_align - T*pc_ref
    if point on T*pc_ref reproject on the correct coordinate camera pixels (K)
    :param pc_ref:
    :param pc_to_align:
    :param T:
    :param indexor:
    :param fact:
    :return:
    '''
    p = kwargs.pop('p', 1)
    factor = kwargs.pop('factor', 1)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    loss = 0
    Q = pc_ref.new_tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])

    for i, pc in enumerate(pc_to_align):
        if inliers is not None:
            pc = torch.cat([point.unsqueeze(1) for n_p, point in enumerate(pc.t()) if inliers[i][n_p]], 1)
            pc_nn = torch.cat([point.unsqueeze(1) for n_p, point in enumerate(pc_ref[i].t()) if inliers[i][n_p]], 1)
        else:
            pc_nn = pc_ref[i]

        pc_nn_t = T[i].matmul(pc_nn) # Should be aligned with pc

        # Repro:
        rep_pc = K[i].matmul(Q.matmul(pc))
        rep_pc_nn_t = K[i].matmul(Q.matmul(pc_nn_t))

        # Get normalized coord:
        n_rep_pc = rep_pc[:2] / rep_pc[2]
        n_rep_pc_nn_t = rep_pc_nn_t[:2] / rep_pc_nn_t[2]

        # Get rounded coord:
        r_n_rep_pc = torch.round(n_rep_pc)
        r_n_rep_pc_nn_t = torch.round(n_rep_pc_nn_t)
        idx = (torch.min(r_n_rep_pc[0, :] == r_n_rep_pc_nn_t[0, :],
                         r_n_rep_pc[1, :] == r_n_rep_pc_nn_t[1, :])).squeeze()

        predicted = rep_pc[2, idx]
        gt =  rep_pc_nn_t[2, idx]
        logger.debug('Selected points {:.2}% (on {:.2}% inliers pruning)'.format(torch.sum(idx).item()/rep_pc.size(1),
                                                                         rep_pc.size(1)/inliers[i].size(0)))
        if torch.sum(idx) < 1:
            logger.warning('No aligned points')
            continue
        elif p == 1:
            loss += func.pairwise_distance(predicted.unsqueeze(0), gt.unsqueeze(0), p=1)
            #loss += func.l1_loss(predicted, gt)
        elif p == 2:
            loss += func.mse_loss(predicted, gt)
        else:
            raise AttributeError('No behaviour for p = {}'.format(p))

    return factor * loss / (i + 1)


def full_pose_loss(predicted, gt, key='full', combine_func=None):

    return combine_func.combine(mean_dist(predicted[key]['p'], gt['p']),
                                mean_dist(predicted[key]['q'], gt['q']))


def matching_loss(pc_ref, pc_to_align, T, inliers=None):
    '''
    Compute the reprojection error of pc_to_align matched to pc_ref according to tf T
    :param pc_ref:
    :param pc_to_align:
    :param T:
    :param indexor:
    :param fact:
    :return:
    '''
    loss = 0
    for i, pc in enumerate(pc_ref):
        if inliers is not None:
            pc = pc[:, inliers[i].byte()]
            #pc = torch.cat([point.unsqueeze(1) for n_p, point in enumerate(pc.t()) if inliers[i][n_p]], 1)
            pc_nn = pc_to_align[i, :, inliers[i].byte()]
            #pc_nn = torch.cat([point.unsqueeze(1) for n_p, point in enumerate(pc_to_align[i].t()) if inliers[i][n_p]], 1)
        else:
            pc_nn = pc_to_align[i]

        #loss += func.mse_loss(T[i].matmul(pc_nn), pc)
        loss += torch.mean(torch.sum((T[i].matmul(pc_nn) - pc) ** 2, 0))

    return loss / (i + 1)


def T_loss(predicted, gt, fact=1):
    eye_mat = predicted.new_zeros(4, 4)
    eye_mat[0, 0] = eye_mat[1, 1] = eye_mat[2, 2] = eye_mat[3, 3] = 1

    loss = 0
    for i, T in enumerate(gt):
        loss += torch.norm(eye_mat - predicted[i].matmul(T.inverse()))

    return fact*loss/(i+1)


def Identity_loss(predicted, fact=1):
    eye_mat = predicted.new_zeros(4, 4)
    eye_mat[0, 0] = eye_mat[1, 1] = eye_mat[2, 2] = eye_mat[3, 3] = 1

    loss = 0
    for i, T in enumerate(predicted):
        loss += torch.norm(eye_mat - T)

    return fact*loss/(i+1)


def minmax_pose_loss(p_ps, p_qs, gt_ps, gt_qs, **kwargs):
    pose_factor = kwargs.pop('pose_factor', 1)
    ori_factor = kwargs.pop('ori_factor', 1)
    min_factor = kwargs.pop('min_factor', 0.1)
    max_factor = kwargs.pop('max_factor', 0.9)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    loss = 0
    for i, p_p in enumerate(p_ps):
        p_loss = pose_factor*mean_dist(p_p, gt_ps[i,:])
        q_loss = ori_factor*mean_dist(p_qs[i, :], gt_qs[i, :])
        loss += max_factor*torch.max(torch.stack((p_loss, q_loss), 0)) + min_factor*torch.min(torch.stack((p_loss, q_loss), 0))

    return loss/(i+1)


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


def exp_triplet_loss(anchor, positives, negatives, **kwargs):
    alpha = kwargs.pop('alpha', 1)
    p = kwargs.pop('p', 2)
    swap = kwargs.pop('swap', True)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    tt_loss = None
    cpt = 0
    for positive in positives:
        for negative in negatives:
            dp = torch.mean(func.pairwise_distance(anchor, positive, p=p))
            dn = torch.mean(func.pairwise_distance(anchor, negative, p=p))
            if swap:
                dn2 = torch.mean(func.pairwise_distance(positive, negative, p=p))
                if dn2 < dn:
                    dn = dn2

            c_loss = torch.log(1 + torch.exp(alpha*(dp-dn)))
            if tt_loss is None:
                tt_loss = c_loss
            else:
                tt_loss += c_loss

            cpt += 1

    tt_loss /= cpt if cpt else 1
    return tt_loss


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
                cpt += 1 if c_loss.item()>0 else 0
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


def reg_loss(map_to_reg, im_ori, **kwargs):
    fact = kwargs.pop('fact', 1e-2)
    reduce_factor = kwargs.pop('reduce_factor', 0.5)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if reduce_factor:
        im_ori = func.interpolate(im_ori, scale_factor=reduce_factor, mode='bilinear', align_corners=True)

    dx_im = torch.exp(-1 * torch.mean(torch.abs(im_ori[:, :, :, :-1] - im_ori[:, :, :, 1:]), dim=1, keepdim=True))
    dy_im = torch.exp(-1 * torch.mean(torch.abs(im_ori[:, :, :-1, :] - im_ori[:, :, 1:, :]), dim=1, keepdim=True))
    dx_mod = torch.abs(map_to_reg[:, :, :, :-1] - map_to_reg[:, :, :, 1:])
    dy_mod = torch.abs(map_to_reg[:, :, :-1, :] - map_to_reg[:, :, 1:, :])

    loss = fact * (torch.sum(dx_im*dx_mod) + torch.sum(dy_im*dy_mod)) / map_to_reg.numel()

    return loss


def image_similarity(predicted_im, gt_im, **kwargs):
    p = kwargs.pop('p', 'L1')
    factor = kwargs.pop('factor', 1)
    no_zeros = kwargs.pop('no_zeros', False)
    alpha = kwargs.pop('alpha', 0.5)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if isinstance(predicted_im, (list, tuple)):
        n_im = len(predicted_im)
        predicted_im = torch.cat(predicted_im, dim=0)
        return image_similarity(predicted_im, gt_im.repeat(n_im, 1, 1, 1), p=p, factor=factor, no_zeros=no_zeros)
    else:
        if no_zeros:
            zeros_idx = torch.max(predicted_im == predicted_im.new_zeros(predicted_im.size()), dim=1)[0].byte()
            zero_mask = predicted_im.new_ones(predicted_im.size())

            if torch.sum(zeros_idx ).item() != 0:
                zero_mask.transpose(1, 3).transpose(1,2)[zeros_idx ] = predicted_im.new_zeros(3)

            gt_im = gt_im*zero_mask

        if p == 'L1':
            loss = factor * func.l1_loss(predicted_im, gt_im)
        elif p == 'L2':
            loss = factor * func.mse_loss(predicted_im, gt_im)
        elif p == 'sum':
            loss = factor * torch.sum(torch.abs(predicted_im - gt_im))
        elif p == 'SSIM':
            ssim = SSIM(predicted_im, gt_im)
            loss = torch.mean(((1 - ssim) / 2).clamp(min=0, max=1))
        elif p == 'mixed':
            ssim = SSIM(predicted_im, gt_im)
            lssim = torch.mean(((1 - ssim) / 2).clamp(min=0, max=1))
            l1 = func.l1_loss(predicted_im, gt_im)
            loss = l1*alpha + lssim *(1-alpha)
        else:
            raise AttributeError('No behaviour for p = {}'.format(p))

    return loss


def l1_modal_loss(predicted_maps, gt_maps, **kwargs):
    p = kwargs.pop('p', 1)
    factor = kwargs.pop('factor', 1)
    listed_maps = kwargs.pop('listed_maps', True)
    no_zeros = kwargs.pop('no_zeros', False)

    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    if listed_maps:
        predicted = torch.cat(predicted_maps, dim=0)
        gt_w_grad = torch.cat(gt_maps, dim=0)
    else:
        predicted = predicted_maps
        gt_w_grad = gt_maps

    gt_w_grad = gt_w_grad.detach()

    if no_zeros:
        if predicted_maps.size(1) > 1:
            zeros_idx = torch.max(predicted_maps == predicted_maps.new_zeros(predicted_maps.size()), dim=1)[0].byte()
            zero_mask = predicted_maps.new_ones(predicted_maps.size())

            if torch.sum(zeros_idx ).item() != 0:
                zero_mask.transpose(1, 3).transpose(1,2)[zeros_idx ] = predicted_maps.new_zeros(3)

            gt = gt_maps*zero_mask
        else:
            gt_w_grad = gt_w_grad.view(1, -1).transpose(0, 1)
            predicted = predicted.view(1, -1).transpose(0, 1)
            non_zeros_idx = gt_w_grad.nonzero()
            gt = gt_w_grad[non_zeros_idx][:, 0]
            predicted = predicted[non_zeros_idx][:, 0]
    else:
        gt = gt_w_grad

    #gt = gt_w_grad.detach()

    if p == 1:
        loss = factor * func.l1_loss(predicted, gt)
    elif p == 2:
        loss = factor * func.mse_loss(predicted, gt)
    elif p == 'sum':
        loss = factor * torch.sum(torch.abs(predicted - gt))
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
