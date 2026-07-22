import math
import sys
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, SingleSampleCenterLoss, MyInfoNCE, MySupervisedInfoNCE,Frameloss
from .backbones import ConvTransformerBackbone, ConvBackbone, CapFormer
from .loc_generators import PointGenerator
from .necks import FPN1D, FPNIdentity, FPNCaPIdentity
sys.path.append("../")
from utils.nms import batched_nms
from old_code.video_audio_encoder import get_model
from video_audio_encoder.VAformer import VAformer, get_weight, load_encoder


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if prior_prob > 0:
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        # fpn_masks remains the same
        return out_logits


class PtTransformerRegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.offset_head = MaskedConv1D(
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), )

        # fpn_masks remains the same
        return out_offsets


class Contextformer(nn.Module):
    """
        Transformer based model for single stage action localization
    """
    def __init__(
        self,
        backbone_type='CapFormer',         # a string defines which backbone we use
        fpn_type='capidentity',              # a string defines which fpn we use
        backbone_arch=(2, 2, 5),         # a tuple defines #layers in embed / stem / branch
        scale_factor=2,          # scale factor between branch layers
        input_dim=0,             # input feat dim
        audio_input_dim=0,    # input aud feat dim
        max_seq_len=768,           # max sequence length (used for training)
        max_buffer_len_factor=6.0, # max buffer size (defined a factor of max_seq_len)
        n_head=4,                # number of heads for self-attention in transformer
        n_mha_win_size=[7, 7, 7, 7, 7, -1],        # window size for self attention; -1 to use full seq  [7, 7, 7, 7, 7, -1]
        n_cap_conv_size=[3, 3, 3, 3, 3, 3],      # win size for cap
        embd_kernel_size=3,  # kernel size of the embedding network
        embd_dim=256,              # output feat channel of the embedding network
        embd_with_ln=True,          # attach layernorm to embedding network
        fpn_dim=256,               # feature dim on FPN
        fpn_with_ln=True,           # if to apply layer norm at the end of fpn
        fpn_start_level=0,       # start level of fpn
        head_dim=256,              # feature dim for head
        regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],      # regression range on each level of FPN
        head_num_layers=3,       # number of layers in the head (including the classifier)
        head_kernel_size=3,      # kernel size for reg/cls heads
        head_with_ln=True,          # attache layernorm to reg/cls heads
        use_abs_pe=False,            # if to use abs position encoding
        use_rel_pe=False,            # if to use rel position encoding
        num_classes=1,           # number of action classes
        k=1.5,                    # K in CaP
        is_affine=True
    ):
        super().__init__()
        input_dim = input_dim + audio_input_dim
        self.multimodal_encoder = get_model(
            pretrained=True, weight='/home/nsccgz/yql/Code/Random_time_length_version/old_code/pretrain_v2.pth'
        )

        # self.multimodal_encoder = load_encoder(
        #     pretrained=True, weight='/home/nsccgz/yql/Code/Random_time_length_version/video_audio_encoder/pretrain_clip_feat.pth'
        # )

        # for p in self.parameters():
        #     p.requires_grad = False

         # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor**i for i in range(
            fpn_start_level, backbone_arch[-1]+1
        )]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes
        self.backbone_arch = backbone_arch

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + backbone_arch[-1])
            self.mha_win_size = n_mha_win_size
        if isinstance(n_cap_conv_size, int):
            self.n_cap_conv_size = [n_cap_conv_size]*(1 + backbone_arch[-1])
        else:
            assert len(n_cap_conv_size) == (1 + backbone_arch[-1])
            self.n_cap_conv_size = n_cap_conv_size

        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = "radius"
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = 1.5
        self.train_reg_loss_weight = 2.0
        self.train_center_loss_weight = 1.0
        self.train_cls_prior_prob = 0.01
        self.train_dropout = 0.0
        self.train_droppath = 0.1
        self.train_label_smoothing = 0.1

        self.center_loss = SingleSampleCenterLoss(D=256)
        self.info_loss = MySupervisedInfoNCE()           # MyInfoNCE  MySupervisedInfoNCE

        # test time config
        self.test_pre_nms_thresh = 0.001
        self.test_pre_nms_topk = 2000
        self.test_iou_threshold = 0.1
        self.test_min_score = 0.001
        self.test_max_seg_num = 100
        self.test_nms_method = 'soft'
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = 0.001
        self.test_multiclass_nms = False
        self.test_nms_sigma = 0.75
        self.test_voting_thresh = 0.9

        self.backbone_type = backbone_type
        self.fpn_type = fpn_type


        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'CapFormer']
        if backbone_type == 'convTransformer':
            self.backbone = ConvTransformerBackbone(
                n_in=input_dim,
                n_embd=embd_dim,
                n_head=n_head,
                n_embd_ks=embd_kernel_size,
                max_len=max_seq_len,
                arch=backbone_arch,
                mha_win_size=self.mha_win_size,
                scale_factor=scale_factor,
                with_ln=embd_with_ln,
                attn_pdrop=0.0,
                proj_pdrop=self.train_dropout,
                path_pdrop=self.train_droppath,
                use_abs_pe=use_abs_pe,
                use_rel_pe=use_rel_pe
            )
        else:
            self.backbone = CapFormer(
                n_in=input_dim,
                n_embd=embd_dim,
                n_embd_ks=embd_kernel_size,
                max_len=max_seq_len,
                arch=backbone_arch,
                scale_factor=scale_factor,
                with_ln=embd_with_ln,
                cap_conv_size=self.n_cap_conv_size,
                k=k,
                use_abs_pe=use_abs_pe,
                path_pdrop=self.train_droppath,
                is_affine=is_affine
            )

            # self.backbone = ConvBackbone(
            #     n_in=input_dim,
            #     n_embd=embd_dim,
            #     n_embd_ks=embd_kernel_size,
            #     arch=backbone_arch,
            #     scale_factor=scale_factor,
            #     with_ln=embd_with_ln
            # )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        # fpn network: convs
        assert fpn_type in ['capidentity', 'identity']
        if fpn_type == 'capidentity':
            self.neck = FPNCaPIdentity(
                in_channels=[embd_dim] * (backbone_arch[-1] + 1),
                out_channel=fpn_dim,
                scale_factor=scale_factor,
                start_level=fpn_start_level,
                with_ln=fpn_with_ln
            )
        else:
            self.neck = FPNIdentity(
                in_channels=[embd_dim] * (backbone_arch[-1] + 1),
                out_channel=fpn_dim,
                scale_factor=scale_factor,
                start_level=fpn_start_level,
                with_ln=fpn_with_ln
            )

        # location generator: points
        self.point_generator = PointGenerator(
            max_seq_len=max_seq_len * max_buffer_len_factor,  # max sequence length that the generator will buffer
            fpn_strides=self.fpn_strides,  # strides of fpn levels
            regression_range=self.reg_range,  # regression range (on feature grids)
        )

        # classfication and regerssion heads
        self.cls_head = PtTransformerClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=[]
        )
        self.reg_head = PtTransformerRegHead(
            fpn_dim, head_dim, len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = 200
        self. loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)

        inp = []

        vid_list = [x['vid_data'].to(self.device).unsqueeze(0) for x in video_list]  # 1 T3 H W
        aud_list = [x['aud_data'].to(self.device).unsqueeze(0) for x in video_list]  # 1 sample

        for vid, aud in zip(vid_list, aud_list):
            v, a = self.multimodal_encoder(vid, aud)  # v:1 C Ti  a:1 C Ti
            inp.append(torch.cat((v.squeeze(0), a.squeeze(0)), dim=0))  # 1536 Ti

            # embeddings = self.multimodal_encoder(vid, aud)  # 1 Ti C
            # inp.append(embeddings.squeeze(0).transpose(1, 0))  # 512 Ti

        batched_inputs, batched_masks = self.preprocessing(inp)

        # forward the network (backbone -> neck -> heads)
        if self.backbone_type == 'convTransformer':
            feats, masks = self.backbone(batched_inputs, batched_masks)
        else:
            feats, masks, gfeats = self.backbone(batched_inputs, batched_masks)

        if self.fpn_type == 'identity':
            fpn_feats, fpn_masks = self.neck(feats, masks)
        else:
            fpn_feats, fpn_masks, fpn_gfeats = self.neck(feats, masks, gfeats)

        # compute the point coordinate along the FPN
        # this is used for computing the GT or decode the final results
        # points: List[T x 4] with length = # fpn levels
        # (shared across all samples in the mini-batch)
        points = self.point_generator(fpn_feats)

        # out_cls: List[B, #cls + 1, T_i]
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        # out_offset: List[B, 2, T_i]
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # permute the outputs
        # out_cls: F List[B, #cls, T_i] -> F List[B, T_i, #cls]
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: F List[B, 2 (xC), T_i] -> F List[B, T_i, 2 (xC)]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: F list[B, 1, T_i] -> F List[B, T_i]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            gt_labels = [x['labels'].to(self.device) for x in video_list]
            gt_frame_labels = [x['frame_labels'].to(self.device) for x in video_list]


            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets, gt_fpn_frame = self.label_points(
                points, gt_segments, gt_labels, gt_frame_labels)

            # compute the loss and return
            if self.fpn_type == 'capidentity' or self.backbone_type == 'CapFormer':
                losses = self.losses2(
                    fpn_masks,
                    out_cls_logits, out_offsets, fpn_feats, fpn_gfeats,
                    gt_cls_labels, gt_offsets, gt_fpn_frame
                )
            else:
                losses = self.losses(
                    fpn_masks,
                    out_cls_logits, out_offsets,
                    gt_cls_labels, gt_offsets
                )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, gt_frame_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        num_levels = len(points)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset, gt_fpn_frame = [], [], []

        # loop over each video sample
        for gt_segment, gt_label, gt_frame_label in zip(gt_segments, gt_labels, gt_frame_labels):
            cls_targets, reg_targets, frame_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label, gt_frame_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)
            gt_fpn_frame.append(frame_targets)

        return gt_cls, gt_offset, gt_fpn_frame

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, gt_frame_label):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)  # 195, n

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)  # 195, n, 2
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)   # 195, n, 2

        if self.train_center_sample == 'radius':
            # center of all segments F T x N
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            # center sampling based on stride radius
            # compute the new boundaries:
            # concat_points[:, 3] stores the stride
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] \
                           - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) \
                            - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack(
                (cb_dist_left, cb_dist_right), -1)
            # F T x N
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0
        else:
            # inside an gt action
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # if there are still more than one actions for one moment
        # pick the one with the shortest duration (easiest to regress)
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        lens.masked_fill_(inside_regress_range==0, float('inf'))
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot   # 195, n  @ n 1
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # OK to use min_len_inds
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride
        reg_targets /= concat_points[:, 3, None]

        # get frame label
        frame_targets = tuple()
        frame_targets += (gt_frame_label, )
        for idx in range(self.backbone_arch[2]-2):
            tmp_frame_label = nn.AvgPool1d(kernel_size=2**(idx+1), stride=2**(idx+1))(gt_frame_label[None])
            tmp_frame_label = torch.where(tmp_frame_label > 0.35,
                                          torch.full_like(tmp_frame_label, 1.),
                                          torch.full_like(tmp_frame_label, 0.))
            frame_targets += (tmp_frame_label.squeeze(0), )

        return cls_targets, reg_targets, frame_targets

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # center_loss = self.center_loss(local_fea, global_fea, gt_frame_labels)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                # 'center_loss': center_loss * 0.2,
                'final_loss' : final_loss}

    def losses2(
        self, fpn_masks,
        out_cls_logits, out_offsets, fpn_feats, fpn_gfeats,
        gt_cls_labels, gt_offsets, gt_fpn_frame
    ):
        # fpn_masks, out_*: F (List) [B, T_i, C]
        # gt_* : B (list) [F T, C]
        # fpn_masks -> (B, FT)
        valid_mask = torch.cat(fpn_masks, dim=1)

        # 1. classification loss
        # stack the list -> (B, FT) -> (# Valid, )
        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        # cat the predicted offsets -> (B, FT, 2 (xC)) -> # (#Pos, 2 (xC))
        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask]
        gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # update the loss normalizer
        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls is already one hot encoded now, simply masking out
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # focal loss
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_reg_loss_weight > 0:
            loss_weight = self.train_reg_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # center_loss = self.center_loss(local_fea, global_fea, gt_frame_labels)

        info_loss = self.info_loss(fpn_feats, fpn_gfeats, gt_fpn_frame)
        # info_loss = self.info_loss(fpn_feats, gt_fpn_frame)
        # frame_loss = self.frame_loss(fpn_scores, gt_fpn_frame)


        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight + info_loss * 0.5
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                # 'center_loss': center_loss * 0.2,
                'info_loss'  : info_loss,
                # 'frame_loss' : frame_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        # video_list B (list) [dict]
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [B, T_i, C]
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        # points F (list) [T_i, 4]
        # fpn_masks, out_*: F (List) [T_i, C]
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            # sigmoid normalization for output logits
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. gather predicted offsets
            offsets = offsets_i[pt_idxs]
            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            
            # 4: repack the results
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results

