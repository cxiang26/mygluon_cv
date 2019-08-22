
from mxnet.gluon import nn

class SpatialGroupEnhance(nn.HybridBlock):
    def __init__(self, groups):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        with self.name_scope():
            self.avg_pool = []
            for i in range(self.groups):
                self.avg_pool.append(nn.GlobalAvgPool2D())
            self.weight = self.params.get(name='weight', shape=(1, groups, 1, 1))
            self.bias = self.params.get(name='bias', shape=(1, groups, 1, 1))
            # self.sig = nn.Activation('sigmoid')
    def hybrid_forward(self, F, x, weight, bias):
        x = x.split(self.groups, axis=1)
        weight = weight.split(self.groups, axis=1)
        bias = bias.split(self.groups, axis=1)
        results = []
        for i, data, w, b in zip(range(self.groups), x, weight, bias):
            data = data * self.avg_pool[i](data).broadcast_like(data)
            data1 = F.sum(data, axis=1, keepdims=True)
            t = data1.reshape((0,-1))
            t = t - t.mean(axis=1, keepdims=True).broadcast_like(t)
            std = F.sqrt(F.mean(F.sum(F.square(t), axis=1, keepdims=True), axis=1, keepdims=True)).broadcast_like(t)
            t = t / (std+1e-5)
            t = t.reshape_like(data1)
            t = t * w.broadcast_like(t) + b.broadcast_like(t)
            result = data * F.sigmoid(t).broadcast_like(data)
            results.append(result)
        results = F.Concat(*results, dim=1)
        return results
