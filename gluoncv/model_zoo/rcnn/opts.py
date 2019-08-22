"""RCNN Model."""
from __future__ import absolute_import

import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd

class capsDense(nn.HybridBlock):
    def __init__(self, dim_c=8, lbl_num=21, input_dim=512, batch_size=128, name='capsnet', eps=1e-8):
        super(capsDense, self).__init__()
        self.dim_c = dim_c
        self.lbl_num = lbl_num
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stddev = 1. / self.input_dim
        self.eps = eps
        with self.name_scope():
            self.w = self.params.get(name='W_'+name, shape=(self.lbl_num, self.input_dim, self.dim_c), init=mx.init.Normal(self.stddev))

    def hybrid_forward(self, F, x, w):
        self.batch_size = x.shape[0]
        x = x.reshape((0, 1, -1, self.input_dim))
        sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
        sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))

        w_out = F.linalg_gemm2(w, sigma)
        w_out = F.expand_dims(w_out, axis=0)
        w_out = F.tile(w_out, reps=(self.batch_size, 1, 1, 1))
        inputs_1 = F.tile(x, (1, self.lbl_num, 1, 1))
        inputs_l = F.linalg_gemm2(inputs_1, w_out)
        inputs_r = F.linalg_gemm2(inputs_l, w.expand_dims(axis=0).tile(reps=(self.batch_size,1,1,1)),transpose_a=False, transpose_b=True)
        output = F.mean(F.sum(inputs_r * inputs_1, axis=-1), axis=-1)
        output = F.sqrt(F.relu(output)).squeeze()
        output = output - output.mean(axis=1, keepdims=True)
        #L2Norm
        #output = output/output.sum(axis=-1, keepdims=True)
        return output

    # def hybrid_forward(self, F, x, w):
    #     self.batch_size = x.shape[0]
    #     x = x.reshape((0, 1, -1, self.input_dim))
    #     sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
    #     sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))
    #
    #     w_out = F.linalg_gemm2(w, sigma)
    #     w_out = F.linalg_gemm2(w_out, w, transpose_a=False, transpose_b=True)
    #     w_out = F.expand_dims(w_out, axis=0)
    #     w_out = F.tile(w_out, reps=(self.batch_size, 1, 1, 1))
    #     inputs_1 = F.tile(x, (1, self.lbl_num, 1, 1))
    #     inputs_ = F.linalg_gemm2(inputs_1, w_out)
    #     output = F.sum(inputs_ * inputs_1, axis=-1)
    #     output = F.sqrt(F.mean(output, axis=-1))
    #     output = output - output.mean(axis=1, keepdims=True)
    #     return output

