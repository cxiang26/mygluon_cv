# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument
"""PeleeNet, implemented in Gluon."""
__all__ = ['PeleeNet', 'peleenet', 'get_peleenet']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

class Conv_bn_relu(HybridBlock):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, gamma_initializer=None, use_relu=True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.HybridSequential()
            if gamma_initializer is not None:
                self.convs.add(
                    nn.Conv2D(in_channels=inp, channels=oup, kernel_size=kernel_size, strides=stride,padding=pad, use_bias=False),
                    nn.BatchNorm(in_channels=oup, gamma_initializer=gamma_initializer),
                    nn.Activation('relu')
                )
            else:
                self.convs.add(
                    nn.Conv2D(in_channels=inp, channels=oup, kernel_size=kernel_size, strides=stride, padding=pad,
                              use_bias=False),
                    nn.BatchNorm(in_channels=oup),
                    nn.Activation('relu')
                )

        else:
            self.convs = nn.HybridSequential()
            if gamma_initializer is not None:
                self.convs.add(
                    nn.Conv2D(channels=oup, kernel_size=kernel_size, strides=stride, padding=pad, use_bias=False),
                    nn.BatchNorm(in_channels=oup, gamma_initializer=gamma_initializer)
                )
            else:
                self.convs.add(
                    nn.Conv2D(channels=oup, kernel_size=kernel_size, strides=stride, padding=pad, use_bias=False),
                    nn.BatchNorm(in_channels=oup)
                )
    def hybrid_forward(self, F, x):
        out = self.convs(x)
        return out

class StemBlock(HybridBlock):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)
        self.stem_2a = Conv_bn_relu(num_init_features, int(num_init_features/2), 1, 1, 0, gamma_initializer='zeros')
        self.stem_2b = Conv_bn_relu(int(num_init_features/2), num_init_features, 3, 2, 1, gamma_initializer='zeros')
        # self.stem_2p = nn.MaxPool2D(pool_size=2, strides=2, pooling_convention='full')
        self.stem_3 = Conv_bn_relu(num_init_features*2, num_init_features, 1, 1, 0)

    def hybrid_forward(self, F, x):
        stem_1_out = self.stem_1(x)
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        # stem_2p_out = self.stem_2p(stem_1_out)
        stem_2p_out = F.Pooling(stem_1_out, kernel=(2,2), stride=(2,2), pool_type='max', pooling_convention='full')
        out = self.stem_3(F.concat(stem_2b_out, stem_2p_out, dim=1))
        return out

class DenseBlock(HybridBlock):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0, gamma_initializer='zeros')
        self.cb1_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1, gamma_initializer='zeros')
        self.cb2_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0, gamma_initializer='zeros')
        self.cb2_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1, gamma_initializer='zeros')
        self.cb2_c = Conv_bn_relu(growth_rate, growth_rate, 3, 1, 1, gamma_initializer='zeros')

    def hybrid_forward(self, F, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = F.concat(x, cb1_b_out, cb2_c_out, dim=1)
        return out

class TransitionBlock(HybridBlock):
    def __init__(self, inp, oup, with_pooling = True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.HybridSequential()
            self.tb.add(Conv_bn_relu(inp, oup, 1, 1, 0),
                        nn.AvgPool2D(pool_size=2, strides=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1, 1, 0)

    def hybrid_forward(self, F, x):
        out = self.tb(x)
        return out

class PeleeNet(HybridBlock):
    def __init__(self, num_classes=1000, num_init_features=32, growthRate=32, nDenseBlocks=[3,4,8,6], bottleneck_width=[1,2,4,4]):
        super(PeleeNet, self).__init__()

        self.num_classes = num_classes
        self.num_init_features = num_init_features

        inter_channel = list()
        total_filter = list()
        dense_inp = list()
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.half_growth_rate = int(growthRate/2)

            self.features.add(StemBlock(3, num_init_features))

            for i, b_w in enumerate(bottleneck_width):
                inter_channel.append(int(self.half_growth_rate * b_w /4) * 4)
                if i == 0:
                    total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                    dense_inp.append(self.num_init_features)
                else:
                    total_filter.append(total_filter[i-1] + growthRate * nDenseBlocks[i])
                    dense_inp.append(total_filter[i-1])

                if i == len(nDenseBlocks) - 1:
                    with_pooling = False
                else:
                    with_pooling = True

                self.features.add(self._make_dense_transition(dense_inp[i], total_filter[i], inter_channel[i],
                                                           nDenseBlocks[i], i+1, with_pooling=with_pooling))

            self.output = nn.HybridSequential()
            self.output.add(nn.Dropout(.5),
                                nn.Dense(units=self.num_classes)) #in_units=total_filter[len(nDenseBlocks)-1]

    def _make_dense_transition(self, dense_inp, total_filter, inter_channel, ndenseblocks, stage_index, with_pooling=True):
        layers = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layers.name_scope():
            for i in range(ndenseblocks):
                layers.add(DenseBlock(dense_inp, inter_channel, self.half_growth_rate))
                dense_inp += self.half_growth_rate * 2

            layers.add(TransitionBlock(dense_inp, total_filter, with_pooling))
        return layers

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = F.Pooling(x, pool_type='avg',kernel=(7,7))
        x = x.flatten()
        x = self.output(x)
        out = F.log_softmax(x, axis=1)
        return out

def get_peleenet(num_layers=50, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False, **kwargs):
    assert num_layers in peleenet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(peleenet_spec.keys()))

    num_init_features, growth_rate, block_config = peleenet_spec[num_layers]
    net = PeleeNet(num_classes=1000, num_init_features=num_init_features, growthRate=growth_rate,
                   nDenseBlocks=block_config, bottleneck_width=[1, 2, 4, 4], **kwargs)
    if pretrained:
        from .model_store import get_model_file
        if not use_se:
            net.load_parameters(get_model_file('resnet%d'%(num_layers),
                                               tag=pretrained, root=root), ctx=ctx)
        else:
            net.load_parameters(get_model_file('se_resnet%d'%(num_layers),
                                               tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net

peleenet_spec = {50:(32, 32, [3, 4, 8, 6])}

def peleenet(**kwargs):
    return get_peleenet(**kwargs)

if __name__ == '__main__':
    import mxnet as mx
    p = get_peleenet(num_layers=50, pretrained=False, ctx=cpu(),
               root='~/.mxnet/models', use_se=False)
    p.initialize(ctx=mx.gpu())
    # for n in p.collect_params().values():
    #     print(n.name)
    # print(p)
    # p.hybridize()
    input = mx.nd.random_uniform(shape=(1,3,750,1333),ctx=mx.gpu())
    # for n in p.features:
    #     input = n(input)
    #     print(input.shape)
    output = p(input)
    p.summary(input)
    # print(output.shape)