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
"""PointNet, implemented in Gluon."""
__all__ = ['PointNetCls', 'PointNetDenseCls', 'pointnetcls', 'pointnetseg']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import nd

# Net
class STN3d(HybridBlock):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        with self.name_scope():
            self.STN3d = nn.HybridSequential(prefix='')
            with self.STN3d.name_scope():
                self.STN3d.add(nn.Conv1D(64, 1), nn.BatchNorm(in_channels=64), nn.Activation('relu'),
                               nn.Conv1D(128, 1), nn.BatchNorm(in_channels=128), nn.Activation('relu'),
                               nn.Conv1D(1024, 1), nn.BatchNorm(in_channels=1024), nn.Activation('relu'),
                               nn.MaxPool1D(num_points), nn.Flatten(),
                               nn.Dense(512), nn.BatchNorm(in_channels=512), nn.Activation('relu'),
                               nn.Dense(256), nn.BatchNorm(in_channels=256), nn.Activation('relu'),
                               nn.Dense(9))
                # self.conv1 = nn.Conv1D(64, 1)
                # self.bn1 = nn.BatchNorm(in_channels=64)
                # self.relu1 = nn.Activation('relu')
                # self.conv2 = nn.Conv1D(128, 1)
                # self.bn2 = nn.BatchNorm(in_channels=128)
                # self.relu2 = nn.Activation('relu')
                # self.conv3 = nn.Conv1D(1024, 1)
                # self.bn3 = nn.BatchNorm(in_channels=1024)
                # self.relu3 = nn.Activation('relu')
                # self.mp1 = nn.MaxPool1D(num_points)
                # self.fla = nn.Flatten()
                # self.fc1 = nn.Dense(512)
                # self.bn4 = nn.BatchNorm(in_channels=512)
                # self.relu4 = nn.Activation('relu')
                # self.fc2 = nn.Dense(256)
                # self.bn5 = nn.BatchNorm(in_channels=256)
                # self.relu5 = nn.Activation('relu')
                # self.fc3 = nn.Dense(9)
            self.iden = self.params.get_constant('iden', value=nd.array([1,0,0,0,1,0,0,0,1],dtype='float32').reshape(1,9))

    def hybrid_forward(self, F, x, iden):

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.mp1(x)
        # x = x.flatten()
        #
        # x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.bn5(self.fc2(x)))
        # x = self.fc3(x)
        x = self.STN3d(x)
        # x = x + iden
        x = F.broadcast_add(x, iden)
        x = F.reshape(x,(-1, 3, 3))
        return x

class PointNetfeat(HybridBlock):
    def __init__(self, num_points = 2500, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.mp1 = nn.MaxPool1D(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def hybrid_forward(self, F, x):
        #
        # if self.routing is not None:
        #     routing_weight = nd.softmax(nd.zeros(shape=(1, 1, self.num_points), ctx=x.context),axis=2)
        trans = self.stn(x)
        x = F.transpose(x,(0,2,1))
        x = F.batch_dot(x, trans)
        x = F.transpose(x,(0,2,1))
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        # if self.routing is not None:
        #     s = nd.sum(x * routing_weight, axis=2, keepdims=True)
        #     # v = Squash(s, axis=1)
        #     for _ in range(self.routing):
        #         routing_weight = routing_weight + nd.sum(x * s, axis=1,keepdims=True)
        #         c = nd.softmax(routing_weight, axis=2)
        #         s = nd.sum(x * c, axis=2, keepdims=True)
        #         # v = Squash(s, axis=1)
        #     x = s
        # else:
        #     x = self.mp1(x)
        if self.global_feat:
            return x, trans
        else:
            x = x.repeat(self.num_points, axis=2)
            return F.concat(x, pointfeat, dim=1), trans

class PointNetfeat_sim(HybridBlock):
    def __init__(self, num_points = 2500, global_feat = True):
        super(PointNetfeat_sim, self).__init__()
        self.k = 30
        self.stn = STN3d(num_points = num_points)
        self.sim = nn.Dense(16, flatten=False)
        self.sim_bn = nn.BatchNorm(in_channels=16)
        self.sim_t = nn.Dense(16, flatten=False)
        self.sim_tbn = nn.BatchNorm(in_channels=16)
        self.conv1 = nn.Conv1D(64, 1)
        self.conv2 = nn.Conv1D(128, 1)
        self.conv3 = nn.Conv1D(1024, 1)
        self.bn1 = nn.BatchNorm(in_channels=64)
        self.bn2 = nn.BatchNorm(in_channels=128)
        self.bn3 = nn.BatchNorm(in_channels=1024)
        self.mp1 = nn.MaxPool1D(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def hybrid_forward(self, F, x):
        trans = self.stn(x)
        x = F.transpose(x,(0,2,1))
        x = F.batch_dot(x, trans)

        sim_mat = F.batch_dot(self.sim(x), self.sim_t(x).transpose((0,2,1)))
        mask = F.topk(sim_mat, ret_typ='value', k=self.k)
        sim_mat = sim_mat * mask
        x = F.transpose(x, (0,2,1))
        x = F.batch_dot(x, F.softmax(sim_mat, axis=1, temperature=0.1))
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        if self.global_feat:
            return x, trans
        else:
            x = x.repeat(self.num_points, axis=2)
            return F.concat(x, pointfeat, dim=1), trans

class PointNetCls(HybridBlock):
    def __init__(self, num_points = 2500, classes = 2, drop_rate=None,routing=None):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.dp_rate = drop_rate
        with self.name_scope():
            self.feat = PointNetfeat(num_points, global_feat=True)
            self.fc1 = nn.Dense(512)
            if drop_rate is not None:
                self.dp = nn.Dropout(drop_rate)
            self.fc2 = nn.Dense(256)
            self.fc3 = nn.Dense(classes)
            self.bn1 = nn.BatchNorm(in_channels=512)
            self.bn2 = nn.BatchNorm(in_channels=256)

    def hybrid_forward(self, F, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        if self.dp_rate is not None:
            x = self.dp(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        # return nd.log_softmax(x, axis=-1), trans
        return x, trans

class PointNetDenseCls(HybridBlock):
    def __init__(self, num_points = 2500, classes = 2, routing=None):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        with self.name_scope():
            self.feat = PointNetfeat(num_points, global_feat=False, routing=routing)
            self.conv1 = nn.Conv1D(512, 1)
            self.conv2 = nn.Conv1D(256, 1)
            self.conv3 = nn.Conv1D(128, 1)
            self.conv4 = nn.Conv1D(classes, 1)
            self.bn1 = nn.BatchNorm(in_channels=512)
            self.bn2 = nn.BatchNorm(in_channels=256)
            self.bn3 = nn.BatchNorm(in_channels=128)

    def hybrid_forward(self, F, x):
        # batchsize = x.shape[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose((0,2,1))
        # x = x.log_softmax(axis=-1)
        # x = x.reshape(batchsize, self.num_points, self.k)
        return x, trans

# Constructor
def pointnetcls(**kwargs):

    net = PointNetCls(**kwargs)
    return net

def pointnetseg(**kwargs):

    net = PointNetDenseCls(**kwargs)
    return net
