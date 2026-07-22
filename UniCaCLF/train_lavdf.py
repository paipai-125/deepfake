import os
gpu_id = '3'   #不能出现空格
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
import argparse
import torch
import numpy as np
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
import random
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from collections import OrderedDict
from typing import List, Union
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from dataset import PretrainFeatDataset_lavdf, PretrainFeatDataset_lavdf_single_modal
from train_utils import trivial_batch_collator, AverageMeter, AP, AR
# from context_aware_model import Cal_model
from contextformer.context_archs import Contextformer

# 自定义解析函数
def parse_comma_separated_list(input_str):
    return input_str.split(',')

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif value.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    Device = 'cuda' if torch.cuda.is_available() else 'cpu'

    conf = argparse.ArgumentParser()
    conf.add_argument("--train_data_root", type=str, default=None,
                      help="The root folder of training set.")
    conf.add_argument("--test_data_root", type=str, default=None,
                      help="The root folder of testing set.")
    conf.add_argument("--train_split", type=parse_comma_separated_list, default=None,
                      help="The used training data.")
    conf.add_argument("--test_split", type=parse_comma_separated_list, default=None,
                      help="The used testing data.")
    conf.add_argument("--feat_folder", type=str, default=None,
                      help="The pretrained video features.")
    conf.add_argument("--audio_feat_folder", type=str, default=None,
                      help="The pretrained audio features.")
    conf.add_argument("--json_file", type=str, default=None,
                      help="The path of json file.")
    conf.add_argument("--is_MultiGPU", type=int, default=0)
    conf.add_argument("--model_name", type=str, default='TVIL_pretrain_feat_v1')
    conf.add_argument("--num_class", type=int, default=1, help='The class number of training dataset')
    conf.add_argument('--lr', type=float, default=1e-4, help='The initial learning rate.')
    conf.add_argument('--epoches', type=int, default=100, help='The training epoches.')
    conf.add_argument('--retrain_epoches', type=int, default=0, help='The start epoch index of retrain.')
    conf.add_argument('--batch_size', type=int, default=8, help='The training batch size over all gpus.')
    conf.add_argument('--image_size', type=int, default=128, help='The image size for training.')
    conf.add_argument("--resume", type=int, default=0)
    conf.add_argument("--resume_model_name", type=str, default='')
    conf.add_argument("--resume_model_name_index", type=str, default='')
    conf.add_argument("--istrain", type=int, default=1)
    conf.add_argument("--istest", type=int, default=1)
    conf.add_argument("--istsne", type=int, default=0)
    conf.add_argument('--max_seq_len', type=int, default=768, help='The max time length for datasets.')
    conf.add_argument('--input_vid_dim', type=int, default=None, help='The channel dim of the input video feature.')
    conf.add_argument('--input_aud_dim', type=int, default=None, help='The channel dim of the audio video feature.')
    conf.add_argument('--type', type=str, default='audio-video',
                      help='The type of model. e.g., audio-video, single-audio, single-video')
    conf.add_argument('--is_multimodal', type=str2bool, default=False, help='If use Sepearte_LayerNorm')
    conf.add_argument('--is_pretrained_fea', type=str2bool, default=True, help='If use pretrained feature extractor')

    args = conf.parse_args()

    time = datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')
    summary_path = os.path.join('summary/weight/', args.model_name)
    tsne_figure_path = os.path.join('summary/figure/', args.model_name)
    resume_path = os.path.join(os.path.join('summary/weight/', args.resume_model_name), args.resume_model_name_index)
    log_path = os.path.join('summary/log/', args.model_name)

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    if not os.path.exists(tsne_figure_path):
        os.makedirs(tsne_figure_path)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists('summary/result/' + args.model_name + '/' + time):
        os.makedirs('summary/result/' + args.model_name + '/' + time)

    tb_writer = SummaryWriter(os.path.join(log_path, 'logs'))

    train_dataset = PretrainFeatDataset_lavdf(
        is_training=True,
        split=args.train_split,
        feat_folder=args.feat_folder,
        audio_feat_folder=args.audio_feat_folder,
        json_file=args.json_file,
        # modal=args.type
    )
    test_dataset = PretrainFeatDataset_lavdf(
        is_training=False,
        split=args.test_split,
        feat_folder=args.feat_folder,
        audio_feat_folder=args.audio_feat_folder,
        json_file=args.json_file,
        # modal=args.type
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=trivial_batch_collator,
                                               drop_last=True,
                                               num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True,
                                              collate_fn=trivial_batch_collator,
                                              drop_last=False,
                                              num_workers=8)

    # model = Cal_model(input_vid_dim=args.input_vid_dim,
    #                   input_aud_dim=args.input_aud_dim,
    #                   type=args.type,
    #                   is_multimodal=args.is_multimodal,
    #                   is_pretrained_fea=args.is_pretrained_fea,
    #                   max_seq_len=args.max_seq_len
    #                   )

    model = Contextformer(
        input_dim=args.input_vid_dim,
        audio_input_dim=args.input_aud_dim,
    )

    if Device == 'cuda':
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        if args.is_MultiGPU:
            model = nn.DataParallel(model, device_ids=[i for i in range(math.ceil(len(gpu_id) / 2))])
    if args.resume:
        model.load_state_dict(torch.load(resume_path))
        print('load successfully')

        # pre_weight = torch.load(model_path)
        # new_pre_weight = OrderedDict()
        # for k, v in pre_weight.items():
        #     if 'att.0' in k:
        #         k = k.replace('att.0', 'att')
        #         new_pre_weight[k] = v
        #     else:
        #         new_pre_weight[k] = v
        # model.load_state_dict(new_pre_weight)

    AP_compute = AP([0.5, 0.75, 0.95])
    AR_compute = AR([100, 50, 30, 20, 10, 5], parallel=False)

    # param1 = []
    # param2 = []
    # param3 = []

    # for name, para in model.named_parameters():
    #     if para.requires_grad == True:
    #         if 'reghead' in name:
    #             param2.append(para)
    #         elif 'video_audio_encoder'in name:
    #             param3.append(para)
    #         else:
    #             param1.append(para)

    # for name, para in model.named_parameters():
    #     if 'video_audio_encoder' in name:
    #         param1.append(para)
    #     else:
    #         param2.append(para)

    # optimizer = optim.Adam([
    #     {'params': param1, 'lr': 0.00005},
    #     {'params': param2, 'lr': 0.001}
    # ])

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, betas=(0.9, 0.999), eps=1e-08)


    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    print(time, 'train:', args.istrain, 'test:', args.istest, ' ==>', args.model_name, '\n', args.image_size, ' begin\n')
    for epoch in range(args.epoches):
        if args.istrain:

            print('Epoch {}/{}'.format(epoch + 1, args.epoches))
            print('-' * 100)
            losses_tracker = {}

            model.train()
            count = 0
            for data_dict in tqdm(train_loader):
                count += 1

                optimizer.zero_grad()

                losses = model(data_dict)
                losses['final_loss'].backward()

                optimizer.step()


                # track all losses
                for key, value in losses.items():
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()

                    losses_tracker[key].update(value.item())

                # if count%200 ==0:
                #     print('Loss {:.2f} ({:.2f})\t'.format(
                # losses_tracker['final_loss'].val,
                # losses_tracker['final_loss'].avg
                # ))

            # print to terminal
            block1 = 'Loss {:.2f} ({:.2f})\t'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )

            block2 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block2 += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2]))

            if tb_writer is not None:

                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.avg
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    epoch + 1
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].avg,
                    epoch + 1
                )


        ap_pre = []
        ap_lab = []
        score_pre = []

        result_avg_list = []
        result_vis_list = []
        label_vis_list = []
        confusion_matrix_pre = []
        confusion_matrix_lab = []
        label_dict = {}
        count_dict = {}
        result_fea_list = []
        result_label_list = []
        if args.istest:
            model.eval()
            with torch.no_grad():
                pre_proposals = []
                gt_seg = []
                video_id = []
                pre_frames = []
                for data_dict in tqdm(test_loader):
                    gt_seg += [x['ori_segments'] for x in data_dict]
                    video_id += [x['video_id'] for x in data_dict]

                    res = model(data_dict)
                    pre_proposals += res

                    # result_avg_list.append(avg_res)
                    # result_vis_list.append(vis_res.cpu().numpy())
                    # label_vis_list.append(gt_fpn.cpu().numpy())


                    # for index, x in enumerate(data_dict):
                    #     for key, value in x.items():
                    #         if key != 'img_data' and key != 'aud_data':
                    #             print("{}: {}".format(key, value))
                    #     print(res[index])

                ap_res = AP_compute(pre_proposals, gt_seg)
                ar_res = AR_compute(pre_proposals, gt_seg)
                print(ap_res)
                print(ar_res)

                epoch_acc = ap_res[0.95].item()

                # result_avg_list = torch.cat(result_avg_list, dim=0)
                # print(result_avg_list.mean(dim=0))

                if args.istsne:
                    all_fea = np.concatenate(result_vis_list, axis=0).transpose(0, 2, 1)  # B T C

                    all_label = np.concatenate(label_vis_list, axis=0)  # B T

                    for index in range(200):  #all_label.shape[0]

                        result_fea = all_fea[index, :, :]
                        result_label = all_label[index, :]

                        X_embedded = TSNE(n_components=2, perplexity=50, learning_rate=500, early_exaggeration=5,
                                          init="pca").fit_transform(result_fea)

                        first_class = []
                        second_class = []
                        for i in range(len(result_label)):
                            if result_label[i] == 0:
                                first_class.append(X_embedded[i, :].reshape(1, 2))
                            elif result_label[i] == 1:
                                second_class.append(X_embedded[i, :].reshape(1, 2))


                        first_class = np.concatenate(first_class, axis=0)
                        second_class = np.concatenate(second_class, axis=0)
                        # thrid_class = np.concatenate(thrid_class, axis=0)
                        # forth_class = np.concatenate(forth_class, axis=0)

                        plt.close('all')

                        plt.figure(figsize=(10, 10), dpi=80)
                        plt.subplots()

                        plt.scatter(first_class[:, 0], first_class[:, 1], c='salmon', label='genuine instants', alpha=0.5)
                        plt.scatter(second_class[:, 0], second_class[:, 1], c='skyblue', label='fake instants', alpha=0.5)
                        # plt.scatter(thrid_class[:200, 0], thrid_class[:200, 1], c='limegreen', label='fake-real',
                        #             alpha=0.5)
                        # plt.scatter(forth_class[:, 0], forth_class[:, 1], c='orange', label='real-fake', alpha=0.5)
                        plt.legend()

                        plt.savefig(os.path.join(tsne_figure_path, '3_a/' + str(index) + '.png'))

                # gt_frame = torch.stack(gt_frame, dim=0)
                #
                # for x in pre_proposals:
                #     pre_frames += [x['frame_pre']]
                # pre_frames = torch.stack(pre_frames, dim=0)
                #
                # epoch_acc = get_F1(gt_frame, pre_frames)
                #
                # print('epoch_acc: {:.4f}'.format(epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, os.path.join(summary_path, str(epoch + 1 + args.retrain_epoches) + '.pth'))

                with open('summary/result/' + args.model_name + '/' + time + '/' + str(epoch + 1 + args.retrain_epoches) + '.txt',
                          'a+') as file:
                    file.write('Best test ACC: ' + str(best_acc) + '\n' + '\n')
                    file.write('AP: ' + str(ap_res) + '\n' + 'AR: ' + str(ar_res) + '\n' + '\n')
                    for index, (gt, pre) in enumerate(zip(gt_seg, pre_proposals)):
                        file.write(
                            str(index) + ' ' + str(video_id[index]) + '\n' + 'gt: ' + str(gt[:10]) + '\n' + 'pr: ' + str(
                                pre['segments'][:10]) + '\n' + '\n')
            print('Best test Acc: {:.4f}'.format(best_acc))
            lr_schedule.step(epoch_acc)
            print('\n')

    tb_writer.close()

