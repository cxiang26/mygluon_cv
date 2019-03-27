import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus to use.')
parser.add_argument('--model', type=str, default='resnet18_v1',
                    help='model to use. options are resnet and wrn. default is resnet.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=100,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='period in epoch for learning rate decays. default is 0 (has no effect).')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                    help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are imperative, hybrid')
parser.add_argument('--save-period', type=int, default=10,
                    help='period in epoch of model saving.')
parser.add_argument('--save-dir', type=str, default='params_caps',
                    help='directory of saved models')
parser.add_argument('--resume-from', type=str,
                    help='resume training from the model')
parser.add_argument('--save-plot-dir', type=str, default='.',
                    help='the path to save the history plot')
opt = parser.parse_args()

CapsPro = True
batch_size = opt.batch_size
classes = 10

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
# context = [mx.gpu(2)]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

model_name = opt.model
if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate}
else:
    kwargs = {'classes': classes}

class squash(nn.HybridBlock):
    def __init__(self, axis=1):
        super(squash, self).__init__()
        self.axis = axis
    def hybrid_forward(self, F, x):
        # norm = F.sum(F.square(x), axis=self.axis, keepdims=True)
        # x = norm / (1+norm) / F.sqrt(norm, keepdims=True) * x
        norm = x.norm(axis=self.axis, keepdims=True)
        x = 5*x / (1+norm)
        return x
class CapsProNet(nn.HybridBlock):
    def __init__(self, features, dim_c=8, lbl_num=10, input_dim=256, batch_size=128, name='capsnet', eps=1e-7):
        super(CapsProNet, self).__init__()
        with self.name_scope():
            self.features = features.features
            self.squash = squash(axis=1)
            self.output = capsDens(dim_c, lbl_num, input_dim, batch_size)

    def hybrid_forward(self, F, x):
        output = self.features(x)
        output = self.squash(output)
        output = self.output(output)
        return output

class capsDens(nn.HybridBlock):
    def __init__(self, dim_c=8, lbl_num=10, input_dim=256, batch_size=128, name='capsnet', eps=1e-7):
        super(capsDens, self).__init__()
        self.dim_c = dim_c
        self.lbl_num = lbl_num
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stddev = 0.1
        self.eps = eps
        with self.name_scope():
            self.w = self.params.get(name='W_'+name, shape=(self.lbl_num, self.input_dim, self.dim_c), init=mx.init.Normal(self.stddev))

    def hybrid_forward(self, F, x, w):
        x = x.reshape((0, self.input_dim, -1, 1))
        x = F.transpose(x, (0, 3, 2, 1))
        self.batch_size = x.shape[0]
        sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
        sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))
        w_out = F.linalg_gemm2(w, sigma)
        w_out = F.linalg_gemm2(w_out, w, transpose_a=False, transpose_b=True)
        w_out = F.expand_dims(w_out, axis=0)
        w_out = F.tile(w_out, (self.batch_size, 1, 1, 1))
        inputs_1 = F.tile(x, (1, self.lbl_num, 1, 1))
        inputs_ = F.linalg_gemm2(inputs_1, w_out)
        output = F.sum(inputs_ * inputs_1, axis=-1)
        output = F.sum(output, axis=-1)
        return output

class capsDens1(nn.HybridBlock):
    def __init__(self, dim_c=8, lbl_num=10, input_dim=512, batch_size=128, name='capsnet', eps=1e-7):
        super(capsDens1, self).__init__()
        self.dim_c = dim_c
        self.lbl_num = lbl_num
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.stddev =  0.1# /np.sqrt(self.input_dim)
        self.eps = eps
        with self.name_scope():
            self.w = self.params.get(name='W_'+name, shape=(self.lbl_num, self.input_dim, self.dim_c), init=mx.init.Normal(self.stddev))

    def hybrid_forward(self, F, x, w):
        # x = x.reshape((0, 1, -1, self.input_dim))
        x = F.transpose(x, (0, 2, 3, 1))
        self.batch_size = x.shape[0]
        sigma = F.linalg_gemm2(w, w, transpose_a=True, transpose_b=False)
        sigma = F.linalg_potri(sigma + self.eps*F.eye(self.dim_c))
        w_out = F.linalg_gemm2(w, sigma)
        w_out = F.linalg_gemm2(w_out, w, transpose_a=False, transpose_b=True)
        w_out = F.expand_dims(w_out, axis=0)
        w_out = F.tile(w_out, reps=(self.batch_size, 1, 1, 1))
        inputs_1 = F.tile(x, (1, self.lbl_num, 1, 1))
        inputs_ = F.linalg_gemm2(inputs_1, w_out)
        output = F.linalg_gemm2(inputs_, inputs_1, transpose_b=True)
        output = F.squeeze(output)
        return output

if CapsPro is True:
    features = get_model(model_name, **kwargs)
    net = CapsProNet(features, dim_c=8, lbl_num=classes, input_dim=256, batch_size=opt.batch_size, name='capsnet', eps=1e-7)
else:
    net = get_model(model_name, **kwargs)

if opt.resume_from:
    net.load_parameters(opt.resume_from, ctx = context)
optimizer = 'nag'

save_period = opt.save_period
if opt.save_dir and save_period:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_period = 0

plot_path = opt.save_plot_dir

import os
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file_path = './sq_matrix_resnet18' + '_train.log'
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
fh = logging.FileHandler(log_file_path)
logger.addHandler(fh)
logger.info(opt)
logger.info('Start training from [Epoch {}]'.format(0))


transform_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    metric = mx.metric.Accuracy()
    train_metric = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_history = TrainingHistory(['training-error', 'validation-error'])

    iteration = 0
    lr_decay_count = 0

    best_val_score = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            train_metric.update(label, output)
            name, acc = train_metric.get()
            iteration += 1

        train_loss /= batch_size * num_batch
        name, acc = train_metric.get()
        nd.waitall()
        name, val_acc = test(ctx, val_data)
        train_history.update([1-acc, 1-val_acc])
        train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

        if val_acc > best_val_score:
            best_val_score = val_acc
            net.save_parameters('%s/%.4f-matrix-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        name, val_acc = test(ctx, val_data)
        logger.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
            (epoch, acc, val_acc, train_loss, time.time()-tic))

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            net.save_parameters('%s/matrix-cifar10-%s-%d.params'%(save_dir, model_name, epoch))

    if save_period and save_dir:
        net.save_parameters('%s/matrix-cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
