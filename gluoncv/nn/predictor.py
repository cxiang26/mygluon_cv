# pylint: disable=unused-argument,arguments-differ
"""Predictor for classification/box prediction."""
from __future__ import absolute_import
import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn


class ConvPredictor(HybridBlock):
    """Convolutional predictor.
    Convolutional predictor is widely used in object-detection. It can be used
    to predict classification scores (1 channel per class) or box predictor,
    which is usually 4 channels per box.
    The output is of shape (N, num_channel, H, W).

    Parameters
    ----------
    num_channel : int
        Number of conv channels.
    kernel : tuple of (int, int), default (3, 3)
        Conv kernel size as (H, W).
    pad : tuple of (int, int), default (1, 1)
        Conv padding size as (H, W).
    stride : tuple of (int, int), default (1, 1)
        Conv stride size as (H, W).
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.

    """
    def __init__(self, num_channel, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                 activation=None, use_bias=True, **kwargs):
        super(ConvPredictor, self).__init__(**kwargs)
        with self.name_scope():
            self.predictor = nn.Conv2D(
                num_channel, kernel, strides=stride, padding=pad,
                activation=activation, use_bias=use_bias,
                weight_initializer=mx.init.Xavier(magnitude=2),
                bias_initializer='zeros')

    def hybrid_forward(self, F, x):
        return self.predictor(x)


class FCPredictor(HybridBlock):
    """Fully connected predictor.
    Fully connected predictor is used to ignore spatial information and will
    output fixed-sized predictions.


    Parameters
    ----------
    num_output : int
        Number of fully connected outputs.
    activation : str, optional
        Optional activation after conv, e.g. 'relu'.
    use_bias : bool
        Use bias in convolution. It is not necessary if BatchNorm is followed.

    """
    def __init__(self, num_output, activation=None, use_bias=True, **kwargs):
        super(FCPredictor, self).__init__(**kwargs)
        with self.name_scope():
            self.predictor = nn.Dense(
                num_output, activation=activation, use_bias=use_bias)

    def hybrid_forward(self, F, x):
        return self.predictor(x)

import math
class CapsPredictor(nn.HybridBlock):
    def __init__(self, dim_c=8, anchor_num=4, lbl_num=21, input_dim=256, batch_size=128, stddev=0.1, eps=1e-7, name='caps'):
        super(CapsPredictor, self).__init__()
        self.vgg16_atrous = [512, 1024, 512, 256, 256, 256]
        self.scale = [1, 1, 1, 0.8, 0.5, 0.5]
        # self.peleenet = [512, 704, 512, 512, 256, 256]
        self.dim_c = dim_c

        self.anchor_num = anchor_num
        self.lbl_num = lbl_num
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stddev = 1. / math.sqrt(self.vgg16_atrous[input_dim])
        self.eps = eps
        with self.name_scope():
            self.w = self.params.get(name='W_'+name, shape=(self.anchor_num, self.lbl_num, self.vgg16_atrous[self.input_dim], self.dim_c),
                                     init=mx.init.Uniform(self.stddev))


    def hybrid_forward(self, F, x, w):
        # x = x.reshape((-1, 1, 8, self.input_dim))
        self.batch_size = x.shape[0]
        x = F.reshape(x, (0, 0, -1, 1))
        x = F.transpose(x, (0,3,2,1))
        sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
        sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))

        w_out = F.linalg_gemm2(w, sigma)
        w_out = F.linalg_gemm2(w_out, w, transpose_a=False, transpose_b=True)
        # w_out = F.reshape(w_out, shape=(self.classes, self.peleenet[self.input_dim], self.peleenet[self.input_dim]))
        # w_out = F.tile(w_out, reps=(self.batch_size, 1, 1, 1))
        inputs_1 = F.tile(x, (1, self.lbl_num, 1, 1))
        temp2 = []
        for j in range(self.anchor_num):
            temp = [F.expand_dims(F.linalg_gemm2(inputs_1[i], w_out[j]), axis=0) for i in range(inputs_1.shape[0])]
            # inputs_ = F.linalg_gemm2(inputs_1, w_out)
            if self.batch_size == 1:
                inputs_ = F.sum(temp[0] * inputs_1, axis=-1)
                inputs_ = F.broadcast_sub(inputs_, inputs_.mean(axis=1, keepdims=True))
                temp2.append(inputs_)
            else:
                inputs_ = F.sum(F.concat(*temp, dim=0) * inputs_1, axis=-1)
                inputs_ = F.broadcast_sub(inputs_, inputs_.mean(axis=1, keepdims=True))
                temp2.append(inputs_)
        output = F.concat(*temp2, dim=1)
        # inputs_1 = F.tile(x, (1, self.lbl_num*self.anchor_num, 1, 1))
        # output = F.sum(inputs_ * inputs_1, axis=-1)
        # output = F.sum(F.softmax(output, axis=1), axis=-1)
        return output
