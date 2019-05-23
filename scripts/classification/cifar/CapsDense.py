import mxnet as mx
from mxnet.gluon import nn


class capsDens(nn.HybridBlock):
    def __init__(self, dim_c=8, lbl_num=21, input_dim=512, batch_size=128, name='capsnet', stddev=0.01, eps=1e-4):
        super(capsDens, self).__init__()
        self.dim_c = dim_c
        self.lbl_num = lbl_num
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stddev = stddev
        self.eps = eps
        with self.name_scope():
            self.w = self.params.get(name='W_'+name, shape=(self.lbl_num, self.input_dim, self.dim_c), init=mx.init.Uniform(self.stddev))
            # self.c = self.params.get_constant(name='dim', value=mx.nd.array([self.input_dim]))

    def hybrid_forward(self, F, x, w):
        # x = x.reshape((-1, 1, 1, self.input_dim))
        # self.batch_size = x.shape[0]
        # x = F.clip(x, a_min=-0.2, a_max=0.2)
        # x = x.transpose((0,2,3,1))
        x = x.squeeze(axis=3)
        x = x.transpose((0,2,1))
        sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
        sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))

        w_out = F.linalg_gemm2(w, sigma)
        # caps_out = F.linalg_gemm2(sigma, w, transpose_b=True)
        # caps_out = F.tile(caps_out, (self.batch_size, 1, 1, 1))
        # inputs_c = F.tile(x, (1, self.lbl_num, self.dim_c, 1))
        # caps_out = F.sum(caps_out * inputs_c, axis=-1)
        w_out = F.linalg_gemm2(w_out, w, transpose_a=False, transpose_b=True)
        w_out = F.expand_dims(w_out, axis=0)
        # w_out = F.reshape(w_out, shape=(1, self.lbl_num, self.input_dim, self.input_dim))
        for i in range(w_out.shape[1]):
            input_ = F.linalg_gemm2(x, F.tile(w_out[:,i,:,:],(x.shape[0],1,1)))
            if i == 0:
                output = F.linalg_gemm2(input_, x, transpose_a=False, transpose_b=True)
            else:
                output = F.concat(output,F.linalg_gemm2(input_, x, transpose_a=False, transpose_b=True),dim=1)
        output = output.squeeze()
        # if output.max() == -3.4028235e+38:
        #     print('min:',output.min(), 'max:',output.max())
        output = F.sqrt(output+self.eps)
        # output = F.linalg_gemm2(inputs_, inputs_1, transpose_a=False, transpose_b=True)
        # output = F.squeeze(output)
        return output

class comcapsDens(nn.HybridBlock):
    def __init__(self, dim_c=8, lbl_num=21, input_dim=512, batch_size=128, name='capsnet', eps=1e-4):
        super(comcapsDens, self).__init__()
        self.dim_c = dim_c
        self.lbl_num = lbl_num
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stddev = 1. / input_dim
        self.eps = eps
        with self.name_scope():
            self.w = self.params.get(name='W_'+name, shape=(self.lbl_num, self.input_dim, self.dim_c), init=mx.init.Uniform(self.stddev))
            # self.c = self.params.get_constant(name='dim', value=mx.nd.array([self.input_dim]))

    def hybrid_forward(self, F, x, w):
        # x = x.reshape((-1, 1, 1, self.input_dim))
        self.batch_size = x.shape[0]
        # x = F.clip(x, a_min=-0.2, a_max=0.2)
        x = x.transpose((0,2,3,1))
        x = x.reshape((0, 1, -1, self.input_dim))
        sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
        sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))

        w_out = F.linalg_gemm2(w, sigma)
        # caps_out = F.linalg_gemm2(sigma, w, transpose_b=True)
        # caps_out = F.tile(caps_out, (self.batch_size, 1, 1, 1))
        # inputs_c = F.tile(x, (1, self.lbl_num, self.dim_c, 1))
        # caps_out = F.sum(caps_out * inputs_c, axis=-1)
        w_out = F.linalg_gemm2(w_out, w, transpose_a=False, transpose_b=True)
        # w_out = F.reshape(w_out, shape=(1, self.lbl_num, self.input_dim, self.input_dim))
        w_out = F.expand_dims(w_out, axis=0)
        w_out = F.tile(w_out, reps=(self.batch_size, 1, 1, 1))
        inputs_1 = F.tile(x, (1, self.lbl_num, 1, 1))
        inputs_ = F.linalg_gemm2(inputs_1, w_out)
        output = F.sum(inputs_ * inputs_1, axis=-1)
        output = F.sqrt(F.relu(F.sum(output, axis=-1))+self.eps)
        output = output - output.mean(axis=1, keepdims=True)
        # output = F.linalg_gemm2(inputs_, inputs_1, transpose_a=False, transpose_b=True)
        # output = F.squeeze(output)
        return output