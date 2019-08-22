# -*- coding: utf-8 -*-
"""Fully Convolutional One-Stage Object Detection."""
from __future__ import absolute_import

import os
import warnings

import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn, HybridBlock
# from IPython import embed

from .fcos_target import FCOSBoxConverter
from ...nn.feature import RetinaFeatureExpander

__all__ = ['FCOS', 'get_fcos',
           'fcos_resnet50_v1_coco',]

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
            var = F.mean(F.cast(F.square(centered_x_new),"float32"), axis=-1, keepdims=True)
            var = F.cast(var, "float16")
        else:
            var = F.mean(F.square(centered_x_new), axis=-1, keepdims=True)
        # (N,G,H*W*C//G) -> (N,C,H,W)
        x_new = F.broadcast_div(centered_x_new, F.sqrt(var + self.eps)).reshape_like(x)
        x_new = F.broadcast_add(F.broadcast_mul(x_new, weight),bias)
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

class FCOS(HybridBlock):
    def __init__(self, network, features, num_filters, classes, steps=None,
                 use_p6_p7=True, pretrained=False,   stds=(0.1, 0.1, 0.2, 0.2),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, share_params = False, **kwargs):
        super(FCOS, self).__init__(**kwargs)
        num_layers = len(features) + int(use_p6_p7)*2
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.share_params = share_params
        self._scale = steps[::-1]

        with self.name_scope():
            bias_init = ClsBiasInit(self.classes)
            self.box_converter = FCOSBoxConverter()
            self.features = RetinaFeatureExpander(network=network,
                                     pretrained=pretrained,
                                     outputs=features)
            self.classes_head = nn.HybridSequential()
            self.box_head = nn.HybridSequential()
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.center_predictors = nn.HybridSequential()
            share_cls_params, share_box_params = None, None
            share_cls_pred_params, share_ctr_pred_params, \
            share_box_pred_params = None, None, None
            for i in range(self._num_layers):
                # classes
                cls_head = RetinaHead(share_params=share_cls_params, prefix='cls')
                self.classes_head.add(cls_head)
                share_cls_params = cls_head.get_params()

                # bbox
                box_head = RetinaHead(share_params=share_box_params, prefix='box')
                self.box_head.add(box_head)
                share_box_params = box_head.get_params()

                # classes preds
                cls_pred = ConvPredictor(num_channels=self.classes,
                                         share_params=share_cls_pred_params, bias_init=bias_init)
                self.class_predictors.add(cls_pred)
                share_cls_pred_params = cls_pred.get_params()

                # center preds
                center_pred = ConvPredictor(num_channels=1,
                                            share_params=share_ctr_pred_params, bias_init='zeros')
                self.center_predictors.add(center_pred)
                share_ctr_pred_params = center_pred.get_params()

                # bbox_pred
                bbox_pred = ConvPredictor(num_channels=4,
                                          share_params=share_box_pred_params, bias_init='zeros')
                self.box_predictors.add(bbox_pred)
                share_box_pred_params = bbox_pred.get_params()

                if not self.share_params:
                    share_cls_params, share_box_params = None, None
                    share_cls_pred_params, share_ctr_pred_params, \
                    share_box_pred_params = None, None, None

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
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        features = self.features(x)
        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        center_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.center_predictors)]

        box_preds = [F.flatten(F.exp(F.transpose(bp(feat), (0, 2, 3, 1)))) * sc
                     for feat, bp, sc in zip(features, self.box_predictors, self._scale)]

        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        center_preds = F.concat(*center_preds, dim=1).reshape((0, -1, 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))

        if autograd.is_training():
            return [cls_preds, center_preds, box_preds]
        cls_prob = F.sigmoid(cls_preds)
        center_prob = F.sigmoid(center_preds)
        cls_prob = F.broadcast_mul(cls_prob, center_prob)
        return cls_prob, box_preds

def get_fcos(name, dataset, pretrained=False, ctx=mx.cpu(),
             root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    "return FCOS network"
    net = FCOS(**kwargs)
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

def fcos_resnet50_v1_coco(pretrained=False, pretrained_base=True, **kwargs):
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    return get_fcos('resnet18_v1',
                   features=['layers2_relu11_fwd', 'layers3_relu68_fwd', 'layers4_relu8_fwd'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[8, 16, 32, 64, 128],
                   classes=classes, dataset='coco', pretrained=pretrained,
                   pretrained_base=pretrained_base,
                   fpn=False, **kwargs)