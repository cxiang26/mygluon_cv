# -*- coding: utf-8 -*-
"""Fully Convolutional One-Stage Object Detection."""
from __future__ import absolute_import

import os
import warnings

import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn, HybridBlock
from ...nn.bbox import BBoxClipToImage
# from IPython import embed

from .fcos_target import FCOSBoxConverter
from ...nn.feature import RetinaFeatureExpander
from ...nn.protomask import Protonet

__all__ = ['MaskFCOS', 'get_maskfcos',
           'maskfcos_resnet50_v1_coco',]

class GroupNorm(nn.HybridBlock):
    """
    If the batch size is small, it's better to use GroupNorm instead of BatchNorm.
    GroupNorm achieves good results even at small batch sizes.
    Reference:
      https://arxiv.org/pdf/1803.08494.pdf
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5,
                 multi_precision=False, prefix=None, **kwargs):
        super(GroupNorm, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('{}_weight'.format(prefix), grad_req='write',
                                          shape=(1, num_channels, 1, 1))
            self.bias = self.params.get('{}_bias'.format(prefix), grad_req='write',
                                        shape=(1, num_channels, 1, 1))
        self.C = num_channels
        self.G = num_groups
        self.eps = eps
        self.multi_precision = multi_precision
        assert self.C % self.G == 0

    def hybrid_forward(self, F, x, weight, bias):
        # (N,C,H,W) -> (N,G,H*W*C//G)
        x_new = F.reshape(x, (0, self.G, -1))
        if self.multi_precision:
            # (N,G,H*W*C//G) -> (N,G,1)
            mean = F.mean(F.cast(x_new, "float32"), axis=-1, keepdims=True)
            mean = F.cast(mean, "float16")
        else:
            mean = F.mean(x_new, axis=-1, keepdims=True)
        # (N,G,H*W*C//G)
        centered_x_new = F.broadcast_minus(x_new, mean)
        if self.multi_precision:
            # (N,G,H*W*C//G) -> (N,G,1)
            var = F.mean(F.cast(F.square(centered_x_new), "float32"), axis=-1, keepdims=True)
            var = F.cast(var, "float16")
        else:
            var = F.mean(F.square(centered_x_new), axis=-1, keepdims=True)
        # (N,G,H*W*C//G) -> (N,C,H,W)
        x_new = F.broadcast_div(centered_x_new, F.sqrt(var + self.eps)).reshape_like(x)
        x_new = F.broadcast_add(F.broadcast_mul(x_new, weight), bias)
        return x_new

class ConvPredictor(nn.HybridBlock):
    def __init__(self, num_channels, share_params=None, bias_init=None, **kwargs):
        super(ConvPredictor, self).__init__(**kwargs)
        with self.name_scope():
            if share_params is not None:
                self.conv = nn.Conv2D(num_channels, 3, 1, 1, params=share_params,
                        bias_initializer=bias_init)
            else:
                self.conv = nn.Conv2D(num_channels, 3, 1, 1,
                        weight_initializer=mx.init.Normal(sigma=0.01),
                        bias_initializer=bias_init)

    def get_params(self):
        return self.conv.params

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x

class RetinaHead(nn.HybridBlock):
    def __init__(self, share_params=None, prefix=None, **kwargs):
        super(RetinaHead, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            for i in range(4):
                if share_params is not None:
                    self.conv.add(nn.Conv2D(256, 3, 1, 1, activation='relu',
                        params=share_params[i]))
                else:
                    self.conv.add(nn.Conv2D(256, 3, 1, 1, activation='relu',
                        weight_initializer=mx.init.Normal(sigma=0.01),
                        bias_initializer='zeros'))
                self.conv.add(GroupNorm(num_channels=256, prefix=prefix))

    def set_params(self, newconv):
        for b, nb in zip(self.conv, newconv):
            b.weight.set_data(nb.weight.data())
            b.bias.set_data(nb.bias.data())

    def get_params(self):
        param_list = []
        for opr in self.conv:
            param_list.append(opr.params)
        return param_list

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        return x

@mx.init.register
class ClsBiasInit(mx.init.Initializer):
    def __init__(self, num_class, cls_method="sigmoid", pi=0.01, **kwargs):
        super(ClsBiasInit, self).__init__(**kwargs)
        self._num_class = num_class
        self._cls_method = cls_method
        self._pi = pi

    def _init_weight(self, name, data):
        if self._cls_method == "sigmoid":
            arr = -1 * np.ones((data.size, ))
            arr = arr *  np.log((1 - self._pi) / self._pi)
            data[:] = arr
        elif self._cls_method == "softmax":
            pass

def cor_target(short, scale):
    h, w, = short, short
    fh = int(np.ceil(np.ceil(np.ceil(h / 2) / 2) / 2))
    fw = int(np.ceil(np.ceil(np.ceil(w / 2) / 2) / 2))
    #
    fm_list = []
    for _ in range(len(scale)):
        fm_list.append((fh, fw))
        fh = int(np.ceil(fh / 2))
        fw = int(np.ceil(fw / 2))
    fm_list = fm_list[::-1]
    #
    cor_targets = []
    for i, stride in enumerate(scale):
        fh, fw = fm_list[i]
        cx = mx.nd.arange(0, fw).reshape((1, -1))
        cy = mx.nd.arange(0, fh).reshape((-1, 1))
        sx = mx.nd.tile(cx, reps=(fh, 1))
        sy = mx.nd.tile(cy, reps=(1, fw))
        syx = mx.nd.stack(sy.reshape(-1), sx.reshape(-1)).transpose()
        cor_targets.append(mx.nd.flip(syx, axis=1) * stride)
    cor_targets = mx.nd.concat(*cor_targets, dim=0)
    return cor_targets

class MaskFCOS(HybridBlock):
    def __init__(self, network, features, classes, steps=None, short=600,
                 valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                 num_prototypes=64,
                 use_p6_p7=True, pretrained_base=False,
                 nms_thresh=0.45, nms_topk=400, post_nms=100, share_params = False, **kwargs):
        super(MaskFCOS, self).__init__(**kwargs)
        num_layers = len(features) + int(use_p6_p7)*2
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.share_params = share_params
        self._scale = steps[::-1]
        self.short = short
        self.base_stride = steps[-1]
        self.valid_range = valid_range
        self.k = num_prototypes

        # input size are solid
        self.cor_targets = self.params.get_constant(name='cor_', value=cor_target(self.short, self._scale))

        with self.name_scope():
            bias_init = ClsBiasInit(len(self.classes))
            self.box_converter = FCOSBoxConverter()
            self.cliper = BBoxClipToImage()
            self.features = RetinaFeatureExpander(network=network, pretrained=pretrained_base, outputs=features)
            self.protonet = Protonet([256, 256, 256, 256, self.k])
            self.classes_head = nn.HybridSequential()
            self.box_head = nn.HybridSequential()
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.center_predictors = nn.HybridSequential()
            self.maskcoe_predictors = nn.HybridSequential()
            share_cls_params, share_box_params = None, None
            share_cls_pred_params, share_ctr_pred_params, \
            share_box_pred_params, share_mask_pred_params = None, None, None, None
            for i in range(self._num_layers):
                # classes
                cls_head = RetinaHead(share_params=share_cls_params, prefix='cls_{}'.format(i))
                self.classes_head.add(cls_head)
                share_cls_params = cls_head.get_params()

                # bbox
                box_head = RetinaHead(share_params=share_box_params, prefix='box_{}'.format(i))
                self.box_head.add(box_head)
                share_box_params = box_head.get_params()

                # classes preds
                cls_pred = ConvPredictor(num_channels=len(self.classes),
                                         share_params=share_cls_pred_params, bias_init=bias_init)
                self.class_predictors.add(cls_pred)
                share_cls_pred_params = cls_pred.get_params()

                # center preds
                center_pred = ConvPredictor(num_channels=1,
                                            share_params=share_ctr_pred_params, bias_init='zeros')
                self.center_predictors.add(center_pred)
                share_ctr_pred_params = center_pred.get_params()

                # mask coefficient preds
                maskcoe_pred = ConvPredictor(num_channels=self.k,
                                             share_params=share_mask_pred_params, bias_init='zeros')
                self.maskcoe_predictors.add(maskcoe_pred)
                share_mask_pred_params = maskcoe_pred.get_params()

                # bbox_pred
                bbox_pred = ConvPredictor(num_channels=4,
                                          share_params=share_box_pred_params, bias_init='zeros')
                self.box_predictors.add(bbox_pred)
                share_box_pred_params = bbox_pred.get_params()

                if not self.share_params:
                    share_cls_params, share_box_params = None, None
                    share_cls_pred_params, share_ctr_pred_params, \
                    share_box_pred_params, share_mask_pred_params = None, None, None, None
            # trainable scales
            self.s1 = self.params.get('scale_p1', shape=(1,), differentiable=True,
                    allow_deferred_init=True, init='ones')
            self.s2 = self.params.get('scale_p2', shape=(1,), differentiable=True,
                    allow_deferred_init=True, init='ones')
            self.s3 = self.params.get('scale_p3', shape=(1,), differentiable=True,
                    allow_deferred_init=True, init='ones')
            self.s4 = self.params.get('scale_p4', shape=(1,), differentiable=True,
                    allow_deferred_init=True, init='ones')
            self.s5 = self.params.get('scale_p5', shape=(1,), differentiable=True,
                    allow_deferred_init=True, init='ones')

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, s1, s2, s3, s4, s5, cor_targets):
        """Hybrid forward"""
        scale_params = [s1, s2, s3, s4, s5]
        features = self.features(x)
        masks = self.protonet(features[-1])
        masks = F.relu(masks)
        cls_head_feat = [cp(feat) for feat, cp in zip(features, self.classes_head)]
        box_head_feat = [cp(feat) for feat, cp in zip(features, self.box_head)]
        cls_preds = [F.transpose(F.reshape(cp(feat), (0, 0, -1)), (0, 2, 1))
                     for feat, cp in zip(cls_head_feat, self.class_predictors)]
        center_preds = [F.transpose(F.reshape(bp(feat), (0, 0, -1)), (0, 2, 1))
                     for feat, bp in zip(cls_head_feat, self.center_predictors)]

        box_preds = [F.transpose(F.exp(F.broadcast_mul(s, F.reshape(bp(feat), (0, 0, -1)))), (0, 2, 1)) * sc
                     for s, feat, bp, sc in zip(scale_params, box_head_feat, self.box_predictors, self._scale)]

        maskeoc_preds = [F.transpose(F.reshape(bp(feat), (0, 0, -1)), (0, 2, 1))
                         for feat, bp in zip(box_head_feat, self.maskcoe_predictors)]

        cls_preds = F.concat(*cls_preds, dim=1)
        center_preds = F.concat(*center_preds, dim=1)
        box_preds = F.concat(*box_preds, dim=1)
        maskeoc_preds = F.concat(*maskeoc_preds, dim=1)
        maskeoc_preds = F.tanh(maskeoc_preds)
        # with autograd.pause():
        #     h, w, = self.short, self.short
        #     fh = int(np.ceil(np.ceil(np.ceil(h / 2) / 2) / 2))
        #     fw = int(np.ceil(np.ceil(np.ceil(w / 2) / 2) / 2))
        #     #
        #     fm_list = []
        #     for _ in range(len(self._scale)):
        #         fm_list.append((fh, fw))
        #         fh = int(np.ceil(fh / 2))
        #         fw = int(np.ceil(fw / 2))
        #     fm_list = fm_list[::-1]
        #     #
        #     cor_targets = []
        #     for i, stride in enumerate(self._scale):
        #         fh, fw = fm_list[i]
        #         cx = F.arange(0, fw).reshape((1, -1))
        #         cy = F.arange(0, fh).reshape((-1, 1))
        #         sx = F.tile(cx, reps=(fh, 1))
        #         sy = F.tile(cy, reps=(1, fw))
        #         syx = F.stack(sy.reshape(-1), sx.reshape(-1)).transpose()
        #         cor_targets.append(F.flip(syx, axis=1) * stride)
        #     cor_targets = F.concat(*cor_targets, dim=0)
        box_preds = self.box_converter(box_preds, cor_targets)

        if autograd.is_training():
            return [cls_preds, center_preds, box_preds, masks, maskeoc_preds]

        box_preds = self.cliper(box_preds, x) # box_preds: [B, N, 4]
        cls_prob = F.sigmoid(cls_preds)
        center_prob = F.sigmoid(center_preds)
        cls_prob = F.broadcast_mul(cls_prob, center_prob)
        cls_id = cls_prob.argmax(axis=-1)
        probs = F.pick(cls_prob, cls_id)
        result = F.concat(cls_id.expand_dims(axis=-1), probs.expand_dims(axis=-1), box_preds, maskeoc_preds, dim=-1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(result, overlap_thresh=self.nms_thresh,
                                         topk=self.nms_topk, valid_thresh=0.001,
                                         id_index=0, score_index=1, coord_start=2, force_suppress=False,
                                         in_format='corner', out_format='corner')
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        maskeoc = F.slice_axis(result, axis=2, begin=6, end=6 + self.k)
        return ids, scores, bboxes, maskeoc, masks

def get_maskfcos(name, dataset, pretrained=False, ctx=mx.cpu(),
             root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    "return FCOS network"
    base_name = name
    net = MaskFCOS(base_name, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('fcos', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net

# def fcos_resnet50_v1_coco(pretrained=False, pretrained_base=True, **kwargs):
#     from ..resnet import resnet50_v1
#     from ...data import COCODetection
#     classes = COCODetection.CLASSES
#     pretrained_base = False if pretrained else pretrained_base
#     base_network = resnet50_v1(pretrained=pretrained_base, **kwargs)
#     features = RetinaFeatureExpander(network=base_network,
#                                      pretrained=pretrained_base,
#                                      outputs=['stage2_activation3',
#                                               'stage3_activation5',
#                                               'stage4_activation2'])
#     return get_fcos(name="resnet50_v1", dataset="coco", pretrained=pretrained,
#                     features=features, classes=classes, base_stride=128, short=800,
#                     max_size=1333, norm_layer=None, norm_kwargs=None,
#                     valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
#                     nms_thresh=0.6, nms_topk=1000, save_topk=100)

def maskfcos_resnet50_v1_coco(pretrained=False, pretrained_base=True, **kwargs):
    from ...data import COCOInstance
    classes = COCOInstance.CLASSES
    return get_maskfcos('resnet50_v1',
                   features=['stage2_activation3', 'stage3_activation5', 'stage4_activation2'],
                   classes=classes,
                   steps=[8, 16, 32, 64, 128],
                   short = 740,
                   valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                   nms_thresh=0.45, nms_topk=1000, post_nms=100,
                   dataset='coco', pretrained=pretrained,
                   num_prototypes=32,
                   pretrained_base=pretrained_base, share_params = True, **kwargs)