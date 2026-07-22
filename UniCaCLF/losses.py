import torch
from torch.nn import functional as F
import torch.nn as nn

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_giou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Generalized Intersection over Union Loss (Hamid Rezatofighi et. al)
    https://arxiv.org/abs/1902.09630

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # giou is reduced to iou in our setting, skip unnecessary steps
    loss = 1.0 - iouk

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287

    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class SingleSampleCenterLoss(nn.Module):
    """
    Single Sample Center Loss

    Reference:
    J Li, Frequency-aware Discriminative Feature Learning Supervised by Single-CenterLoss for Face Forgery Detection, CVPR 2021.

    Parameters:
        m (float): margin parameter.
        D (int): feature dimension.
        C (vector): learnable center.
    """

    def __init__(self, m=0.9, D=128, use_gpu=True):
        super(SingleSampleCenterLoss, self).__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.use_gpu = use_gpu
        self.l2loss = nn.MSELoss(reduction='none')


    def forward(self, x, centers, labels):  # x: list num_layer [B C T]  center: list num_layer [B C 1]  labels: list num_sample [num_layer]
        """
        Args:
            x: feature matrix with shape (b, c, t).
            center: (b, c, 1)
            labels: ground truth labels with shape (batch_size, t).
        """

        losses = 0.
        total_losses = 0.
        for j, (feats, global_feas) in enumerate(zip(x, centers)):     # jth layer feature
            if j ==0:
                for i, (feat, center) in enumerate(zip(feats, global_feas)):  # ith sample
                    label = labels[i][j]  # Ti
                    valid_T = len(label)
                    feat = feat[:, :valid_T]   # masking out valid feat  C Ti

                    eud_mat = torch.sqrt(self.l2loss(feat, center.expand(-1, valid_T)).sum(dim=0))  # Ti

                    fake_count = label.sum()
                    real_count = (1 - label).sum()

                    dist_fake = (eud_mat * label.float()).clamp(min=1e-12, max=1e+12).sum()
                    dist_real = (eud_mat * (1 - label.float())).clamp(min=1e-12, max=1e+12).sum()

                    if fake_count != 0:
                        dist_fake /= fake_count
                    if real_count != 0:
                        dist_real /= real_count

                    max_margin = dist_real - dist_fake + self.margin

                    max_margin = max_margin if max_margin > 0 else torch.zeros_like(max_margin).to(max_margin.device)

                    losses += (dist_real + max_margin)

                total_losses += (losses/len(labels))

        return total_losses


class MyInfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', metric='cosine'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.metric = metric

    def forward(self, fpn_feats, gt_fpn_frame):
        info_nec_loss = []
        for j, feats in enumerate(fpn_feats):   # jth layer feature
            if j ==0:
                for i, feat in enumerate(feats):   # ith sample
                    label = gt_fpn_frame[i][j]
                    feat = feat[:, :len(label)].transpose(1, 0)   # T C
                    query_sample = feat[label == 0]
                    negative_sample = feat[label == 1]
                    if negative_sample.size(0) == 0 or query_sample.size(0) == 0:
                        continue
                    info_nec_loss.append(info_nce(query_sample, query_sample, negative_sample,
                                                  temperature=self.temperature,
                                                  reduction=self.reduction,
                                                  negative_mode=self.negative_mode, metric=self.metric))

        batch_loss = torch.mean(torch.stack(info_nec_loss).squeeze())

        return batch_loss


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', metric='cosine'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        # positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        sim_matrix = my_distance(query, positive_key, metric)
        mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)

        positive_logit = torch.mean(sim_matrix, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = my_distance(query, negative_keys, metric)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


class MySupervisedInfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', metric='cosine'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.metric = metric

    def forward(self, fpn_feats, fpn_gfeats, gt_fpn_frame):
        info_nec_loss = []
        for j, (feats, gfeats) in enumerate(zip(fpn_feats, fpn_gfeats)):   # jth layer feature
            if j == 0:                  # warning gt_frame_label
                for i, (feat, global_fea) in enumerate(zip(feats, gfeats)):   # ith sample
                    query_sample = global_fea.transpose(1, 0)
                    label = gt_fpn_frame[i][j]
                    feat = feat[:, :len(label)].transpose(1, 0)   # T C
                    positive_sample = feat[label == 0]
                    negative_sample = feat[label == 1]
                    if negative_sample.size(0) == 0 or positive_sample.size(0) == 0:
                        continue
                    # dict_size = 100  # could be larger according to gpu memory
                    # positive_sample = positive_sample[torch.randperm(positive_sample.size()[0])[:negative_sample.size(0)]]
                    # negative_sample = negative_sample[torch.randperm(negative_sample.size(0))[:dict_size]]
                    info_nec_loss.append(supervised_info_nce(query_sample, positive_sample, negative_sample,
                                                  temperature=self.temperature,
                                                  reduction=self.reduction,
                                                  negative_mode=self.negative_mode, metric=self.metric))

        batch_loss = torch.mean(torch.stack(info_nec_loss).squeeze())

        return batch_loss



def supervised_info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', metric='cosine'):
    # Check input dimensionality.    query: 1 C   psoitive_key: N C   negative_keys: M, C    notice: N + M=T
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")


    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        # positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        sim_matrix = my_distance(query, positive_key, metric)  # 1 N

        positive_logit = torch.mean(sim_matrix, dim=1, keepdim=True)  # 1 1
        # positive_logit = torch.sum(sim_matrix, dim=1, keepdim=True)  # 1 1


        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = my_distance(query, negative_keys, metric)  # 1 M
            # negative_logits = torch.mean(negative_logits, dim=1, keepdim=True)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)  # 1 M+1
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


# class Frameloss(nn.Module):
#
#     def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', metric='cosine'):
#         super().__init__()
#         self.temperature = temperature
#         self.reduction = reduction
#         self.negative_mode = negative_mode
#         self.metric = metric
#         self.bce_loss = nn.BCEWithLogitsLoss()
#
#     def forward(self, fpn_feats, fpn_gfeats, gt_fpn_frame):
#         sim_scores = []
#         for j, (feats, gfeats) in enumerate(zip(fpn_feats, fpn_gfeats)):   # jth layer feature
#             if j <2:                  # warning gt_frame_label
#                 for i, (feat, global_fea) in enumerate(zip(feats, gfeats)):   # ith sample
#                     global_fea = global_fea.transpose(1, 0)   # 1 C
#                     label = gt_fpn_frame[i][j]
#                     feat = feat[:, :len(label)].transpose(1, 0)   # T C
#                     sim_score = my_distance(feat, global_fea, metric=self.metric).squeeze(-1) * -1  # T 1
#                     sim_scores.append(self.bce_loss(sim_score.unsqueeze(0), label.unsqueeze(0).float()))
#
#         batch_loss = torch.mean(torch.stack(sim_scores).squeeze())
#
#         return batch_loss


class Frameloss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fpn_scores, gt_fpn_frame):
        loss = []
        for j, scores in enumerate(fpn_scores):   # jth layer feature
            if j ==0:                  # warning gt_frame_label
                for i, score in enumerate(scores):   # ith sample
                    label = gt_fpn_frame[i][j]
                    loss.append(self.bce_loss(score[:, :len(label)] * -1, label.unsqueeze(0).float()))

        batch_loss = torch.mean(torch.stack(loss).squeeze())

        return batch_loss



def my_distance(x, y, metric='cosine'):
    if metric == 'cosine':
        return torch.mm(x, y.t())
    else:
        return torch.cdist(x, y)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

