import os
import copy
import random
import numpy as np
import random
import torch
from tqdm import tqdm
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import datetime
from torchsummary import summary
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from typing import List, Union
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


class Saver(object):
    def __init__(self, path_figure1):
        self.path_figure1 = path_figure1
        self.figure_saver()

    def figure_saver(self):
        # # drawing tool
        print('---------------------------Drawing...--------------------------')
        epochs = range(1, len(result_train_loss) + 1)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        lns1 = ax1.plot(epochs, result_train_loss, 'b', label="train Loss")
        lns3 = ax1.plot(epochs, result_val_loss, 'k', label="test Loss")

        lns2 = ax2.plot(epochs, result_train_score, 'r', label="train score")
        lns7 = ax2.plot(epochs, result_val_score, 'y', label="final score")

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax2.set_ylabel('f1 score')
        # 合并图例
        lns = lns1 + lns2 + lns3 + lns7
        labels = ["train Loss", "train score", "test Loss", "final score"]
        plt.legend(lns, labels, loc=2)
        plt.savefig(self.path_figure1)
        plt.show()


def F1(y_true, y_pred):
    epsilon = 1e-5

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))  # TP

        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + epsilon)
        # print(possible_positives)
        # print('recall', recall)
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))  # TP
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + epsilon)
        # print(predicted_positives)
        # print('precesion', precision)
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + epsilon))


def get_F1(label, prediction):
    batch = label.shape[0]
    f1 = 0
    for i in range(batch):
        Y_t = label[i, :]
        Y_p = prediction[i, :]
        # Y_p = np.round(Y_p)
        f1 += f1_score(Y_t, Y_p, average='binary')
        # f1 += F1(Y_t.numpy(), Y_p.numpy())
    return f1 / batch


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors."""

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors + (box_max - box_min) - inter_len
    iou = inter_len / union_len
    return iou


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class AP(nn.Module):
#     """
#     Average Precision
#
#     The mean precision in Precision-Recall curve.
#     """
#
#     def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5):
#         super().__init__()
#         self.iou_thresholds: List[float] = iou_thresholds if type(iou_thresholds) is list else [iou_thresholds]
#         self.n_labels = 0
#         self.ap: dict = {}
#         self.proposals = []
#         self.labels = []
#
#     def forward(self, pre_data_dict, gt_list):
#
#         for data in pre_data_dict:
#             score = data['scores'].unsqueeze(-1)
#             seg = data['segments']
#             self.proposals.append(torch.cat((score, seg), dim=-1))
#
#         self.labels = gt_list
#
#         for iou_threshold in self.iou_thresholds:
#             values = []
#             self.n_labels = 0
#
#             for index in range(len(self.labels)):
#                 proposals = self.proposals[index]
#                 # print('proposal size :{}'.format(proposals.size()))
#                 labels = self.labels[index]
#                 # print('labels size :{}'.format(labels.size()))
#
#                 values.append(AP.get_values(iou_threshold, proposals, labels))
#                 self.n_labels += len(labels)
#
#             # sort proposals
#             values = torch.cat(values)
#             ind = values[:, 0].sort(stable=True, descending=True).indices
#             values = values[ind]
#
#             # accumulate to calculate precision and recall
#             curve = self.calculate_curve(values)
#             ap = self.calculate_ap(curve)
#             self.ap[iou_threshold] = ap
#
#         return self.ap
#
#     def calculate_curve(self, values):
#         acc_TP = 0
#         acc_FP = 0
#         curve = torch.zeros((len(values), 2))
#         for i, (confidence, is_TP) in enumerate(values):
#             if is_TP == 1:
#                 acc_TP += 1
#             else:
#                 acc_FP += 1
#
#             precision = acc_TP / (acc_TP + acc_FP)
#             recall = acc_TP / self.n_labels
#             curve[i] = torch.tensor((recall, precision))
#
#         curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
#         return curve
#
#     def calculate_ap(self, curve):
#         y_max = 0.
#         ap = 0
#         for i in range(len(curve) - 1):
#             x1, y1 = curve[i]
#             x2, y2 = curve[i + 1]
#             if y1 > y_max:
#                 y_max = y1
#             dx = x1 - x2
#             ap += dx * y_max
#         return ap
#
#     @staticmethod
#     def get_values(
#             iou_threshold: float,
#             proposals: Tensor,
#             labels: Tensor,
#     ) -> Tensor:
#         n_labels = len(labels)
#         ious = torch.zeros((len(proposals), n_labels))  # 1, 1
#         for i in range(len(labels)):
#             ious[:, i] = iou_with_anchors(proposals[:, 1], proposals[:, 2], labels[i, 0], labels[i, 1])
#
#         # values: (confidence, is_TP) rows
#         n_labels = ious.shape[1]
#         detected = torch.full((n_labels,), False)
#         confidence = proposals[:, 0]
#         potential_TP = ious > iou_threshold
#
#         for i in range(len(proposals)):
#             for j in range(n_labels):
#                 if potential_TP[i, j]:
#                     if detected[j]:
#                         potential_TP[i, j] = False
#                     else:
#                         # mark as detected
#                         potential_TP[i] = False  # mark others as False
#                         potential_TP[i, j] = True  # mark the selected as True
#                         detected[j] = True
#
#         is_TP = potential_TP.any(dim=1)
#         values = torch.column_stack([confidence, is_TP])
#         return values


class AP(nn.Module):
    """
    Average Precision

    The mean precision in Precision-Recall curve.
    """

    def __init__(self, iou_thresholds: Union[float, List[float]] = 0.5):
        super().__init__()
        self.iou_thresholds: List[float] = iou_thresholds if type(iou_thresholds) is list else [iou_thresholds]
        self.n_labels = 0
        self.ap: dict = {}
        self.proposals = []
        self.labels = []

    def forward(self, pre_data_dict, gt_list):
        self.proposals = []
        for data in pre_data_dict:
            score = data['scores'].unsqueeze(-1)
            seg = data['segments']
            self.proposals.append(torch.cat((score, seg), dim=-1))

        self.labels = gt_list
        # print('len labels:{}'.format(len(self.labels)))

        for iou_threshold in self.iou_thresholds:
            values = []
            self.n_labels = 0

            for index in range(len(self.labels)):
                proposals = self.proposals[index]
                # print('proposal size :{}'.format(proposals.size()))
                labels = self.labels[index]

                # with open('summary/result/proposal_show.txt', 'a+') as file:
                #         file.write( str(proposals) + '\n' + str(labels) + '\n' + '\n')

                self.n_labels += len(labels)   # Count the total number of ground truth segments
                if len(proposals) == 0:
                    continue
                # print('labels size :{}'.format(labels.size()))
                values.append(AP.get_values(iou_threshold, proposals, labels))

            # sort proposals
            values = torch.cat(values)
            _, ind = torch.sort(values[:, 0], dim=0, descending=True)
            # ind = values[:, 0].sort(stable=True, descending=True).indices
            values = values[ind]

            # accumulate to calculate precision and recall
            curve = self.calculate_curve(values)
            ap = self.calculate_ap(curve)
            self.ap[iou_threshold] = ap

        return self.ap

    def calculate_curve(self, values):
        is_TP = values[:, 1]
        acc_TP = torch.cumsum(is_TP, dim=0)
        precision = acc_TP / (torch.arange(len(is_TP)) + 1)
        recall = acc_TP / self.n_labels
        curve = torch.stack([recall, precision]).T
        curve = torch.cat([torch.tensor([[1., 0.]]), torch.flip(curve, dims=(0,))])
        return curve

    @staticmethod
    def calculate_ap(curve):
        x, y = curve.T
        y_max = y.cummax(dim=0).values
        x_diff = x.diff().abs()
        ap = (x_diff * y_max[:-1]).sum()
        return ap

    @staticmethod
    def get_values(
            iou_threshold: float,
            proposals: Tensor,
            labels: Tensor,
    ) -> Tensor:
        n_labels = len(labels)
        n_proposals = len(proposals)

        # Compute IoUs between each proposal and each label
        ious = torch.zeros((len(proposals), n_labels))  # (n_proposals, n_labels)
        for i in range(n_labels):
            ious[:, i] = iou_with_anchors(proposals[:, 1], proposals[:, 2], labels[i, 0], labels[i, 1])

        # values: (confidence, is_TP) rows
        # n_labels = ious.shape[1]
        # detected = torch.full((n_labels,), False)
        confidence = proposals[:, 0]
        potential_TP = ious > iou_threshold

        tp_indexes = []

        for i in range(n_labels):
            potential_TP_index = potential_TP[:, i].nonzero()
            for (j,) in potential_TP_index:
                if j not in tp_indexes:
                    tp_indexes.append(j)
                    break
        is_TP = torch.zeros(n_proposals, dtype=torch.bool)
        if len(tp_indexes) > 0:
            tp_indexes = torch.stack(tp_indexes)
            is_TP[tp_indexes] = True
        values = torch.column_stack([confidence, is_TP])
        return values


class AR(nn.Module):
    """
    Average Recall

    Args:
        n_proposals_list: Number of proposals. 100 for AR@100.
        iou_thresholds: IOU threshold samples for the curve. Default: [0.5:0.05:0.95]

    """

    def __init__(self, n_proposals_list: Union[List[int], int] = 100, iou_thresholds: List[float] = None,
                 parallel: bool = True
                 ):
        super().__init__()
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.n_proposals_list: List[int] = n_proposals_list if type(n_proposals_list) is list else [n_proposals_list]
        self.iou_thresholds = iou_thresholds
        self.parallel = parallel
        self.ar: dict = {}
        self.proposals = []
        self.labels = []

    def forward(self, pre_data_dict, gt_list):
        self.proposals = []
        for data in pre_data_dict:
            score = data['scores'].unsqueeze(-1)
            seg = data['segments']
            self.proposals.append(torch.cat((score, seg), dim=-1))

        self.labels = gt_list

        for n_proposals in self.n_proposals_list:
            if self.parallel:
                with ProcessPoolExecutor(cpu_count() // 2 - 1) as executor:
                    futures = []
                    for meta in metadata:
                        proposals = torch.tensor(proposals_dict[meta.file])
                        labels = torch.tensor(meta.fake_periods)
                        futures.append(executor.submit(AR.get_values, n_proposals, self.iou_thresholds,
                                                       proposals, labels, 25.))

                    values = list(map(lambda x: x.result(), futures))
                    values = torch.stack(values)
            else:
                values = torch.zeros((len(self.proposals), len(self.iou_thresholds), 2))
                for index in range(len(self.labels)):
                    proposals = self.proposals[index]
                    labels = self.labels[index]
                    values[index] = AR.get_values(n_proposals, self.iou_thresholds, proposals, labels)

            values_sum = values.sum(dim=0)

            TP = values_sum[:, 0]
            FN = values_sum[:, 1]
            recall = TP / (TP + FN)
            self.ar[n_proposals] = recall.mean()

        return self.ar

    @staticmethod
    def get_values(
            n_proposals: int,
            iou_thresholds: List[float],
            proposals: Tensor,
            labels: Tensor,
    ):
        proposals = proposals[:n_proposals]
        n_proposals = proposals.shape[0]
        n_labels = len(labels)
        ious = torch.zeros((n_proposals, n_labels))
        for i in range(len(labels)):
            ious[:, i] = iou_with_anchors(proposals[:, 1], proposals[:, 2], labels[i, 0], labels[i, 1])

        n_thresholds = len(iou_thresholds)

        # values: rows of (TP, FN)
        iou_max = ious.max(dim=0)[0]
        values = torch.zeros((n_thresholds, 2))

        for i in range(n_thresholds):
            iou_threshold = iou_thresholds[i]
            TP = (iou_max > iou_threshold).sum()
            FN = n_labels - TP
            values[i] = torch.tensor((TP, FN))

        return values