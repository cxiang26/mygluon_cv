from __future__ import absolute_import
from mxnet import gluon


class Protonet(gluon.HybridBlock):
    def __init__(self, channels=[256, 256, 256, 64], act_fun=None, upsampling_ratio=2, **kwargs):
        super(Protonet, self).__init__(**kwargs)

        self.channels = channels
        self.upsampling_ratio=upsampling_ratio
        self.activation = act_fun
        with self.name_scope():
            self.net = gluon.nn.HybridSequential(prefix='protonet_')
            self.mask = gluon.nn.HybridSequential(prefix='mask_')
            for i, channel in enumerate(self.channels):
                if i < len(self.channels)-2:
                    self.net.add(gluon.nn.Conv2D(channels=channel, kernel_size=(3,3), strides=(1,1), padding=(1, 1)))
                    self.net.add(gluon.nn.Activation('relu'))
                    # self.net.add(gluon.nn.BatchNorm())
                elif i == len(self.channels) - 2:
                    self.mask.add(gluon.nn.Conv2DTranspose(in_channels=channel, channels=channel, kernel_size=(2,2),strides=(2,2), groups=channel))
                    self.mask.add(gluon.nn.Conv2D(channels=channel, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)))
                    self.mask.add(gluon.nn.Activation('relu'))
                else:
                    self.mask.add(gluon.nn.Conv2D(channels=channel, kernel_size=(1,1), strides=(1,1), padding=(0, 0), use_bias=False))
                    self.mask.add(gluon.nn.Activation('relu'))
    def hybrid_forward(self, F, x):
        x = self.net(x)
        # x = F.UpSampling(x, scale=self.upsampling_ratio, num_filter=3, sample_type='bilinear')
        x = self.mask(x)
        return x