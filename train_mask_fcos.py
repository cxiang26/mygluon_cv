# -*- coding: utf-8 -*-
"Train FCOS end to end."
import os
import argparse

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.fcos import \
        MaskFCOSDefaultTrainTransform, MaskFCOSDefaultValTransform
from gluoncv.utils.metrics.coco_instance import COCOInstanceMetric
# from IPython import embed

def parse_args():
    parser = argparse.ArgumentParser(description='Train MaskFCOS networks e2e.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=740,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=8, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, '
                                        'if your CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='1,2',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='45',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',#'/mnt/mdisk/xcq/results/mask_fcos/smooth_maskfcos_resnet50_v1_coco_best.params',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='15, 30, 37, 41',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=int, default=0,
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='/mnt/mdisk/xcq/results/mask_fcos/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')

    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true', default=True,
                        help='Whether to use static memory allocation. Memory usage will increase.')

    args = parser.parse_args()
    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '12,16'
        args.lr = float(args.lr) if args.lr else 0.00125
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 8000
        args.wd = float(args.wd) if args.wd else 1e-4
        num_gpus = len(args.gpus.split(','))
        if num_gpus == 1:
            args.lr_warmup = -1
        else:
            args.lr *= num_gpus
            args.lr_warmup /= num_gpus
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        train_dataset = gdata.COCOInstance(root='/home/xcq/PycharmProjects/datasets/coco/',splits='instances_train2017')
        val_dataset = gdata.COCOInstance(root='/home/xcq/PycharmProjects/datasets/coco/',splits='instances_val2017', skip_empty=False)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_workers):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Stack() for _ in range(6)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(train_transform(
            net.short, net.base_stride, net.valid_range)),
            batch_size,shuffle=True, batchify_fn=train_bfn, last_batch='rollover',
            num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Stack() for _ in range(2)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(net.short, net.base_stride)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def get_lr_at_iter(alpha):
    return 1. / 10. * (1 - alpha) + alpha


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch

def crop(bboxes, h, w, masks):
    scale = 4
    b = masks.shape[0]
    with autograd.pause():
        ctx = bboxes.context
        _h = mx.nd.arange(h, ctx=ctx)
        _w = mx.nd.arange(w, ctx = ctx)
        _h = mx.nd.tile(_h, reps=(b, 1))
        _w = mx.nd.tile(_w, reps=(b, 1))
        x1, y1 = mx.nd.round(bboxes[:, 0]/scale), mx.nd.round(bboxes[:, 1]/scale)
        x2, y2 = mx.nd.round((bboxes[:, 2])/scale), mx.nd.round((bboxes[:, 3])/scale)
        _h = (_h >= y1.expand_dims(axis=-1)) * (_h <= y2.expand_dims(axis=-1))
        _w = (_w >= x1.expand_dims(axis=-1)) * (_w <= x2.expand_dims(axis=-1))
        _mask = mx.nd.batch_dot(_h.expand_dims(axis=-1), _w.expand_dims(axis=-1), transpose_b=True)
    masks = _mask * masks
    return masks

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        det_info = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        for x, det_inf in zip(data, det_info):
            # get prediction results
            det_id, det_score, det_bbox, det_maskeoc, det_mask = net(x)
            det_bbox = clipper(det_bbox, x)
            for i in range(det_bbox.shape[0]):
                det_bbox_t = det_bbox[i]
                det_id_t = det_id[i].asnumpy()
                det_score_t = det_score[i].asnumpy()
                det_maskeoc_t = det_maskeoc[i]
                det_mask_t = det_mask[i]
                full_mask = mx.nd.dot(det_maskeoc_t, det_mask_t)
                im_height, im_width, h_scale, w_scale = det_inf[i].asnumpy()
                im_height, im_width = int(round(im_height / h_scale)), int(
                    round(im_width / w_scale))
                full_mask = mx.nd.sigmoid(full_mask)
                _, h, w = full_mask.shape
                full_mask = crop(det_bbox_t, h, w, full_mask).asnumpy()
                det_bbox_t = det_bbox_t.asnumpy()
                det_bbox_t[:, 0], det_bbox_t[:, 2] = det_bbox_t[:, 0] / w_scale, det_bbox_t[:, 2] / w_scale
                det_bbox_t[:, 1], det_bbox_t[:, 3] = det_bbox_t[:, 1] / h_scale, det_bbox_t[:, 3] / h_scale
                full_masks = []
                for mask in full_mask:
                    full_masks.append(gdata.transforms.mask.proto_fill(mask, (im_width, im_height)))
                full_masks = np.array(full_masks)
                eval_metric.update(det_bbox_t, det_id_t, det_score_t, full_masks)
    return eval_metric.get()


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    # net.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(
        net.collect_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': args.momentum})

    # lr_decay_policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)

    # losses and metrics
    # maskfcos_cls_loss = gcv.loss.SigmoidFocalLoss(
    #         from_logits=False, sparse_label=True, num_class=len(net.classes)+1)
    # maskfcos_ctr_loss = gcv.loss.CtrNessLoss()
    # maskfcos_box_loss = gcv.loss.IOULoss(return_iou=False)
    # maskfcos_mask_loss = gcv.loss.MaskLoss()
    maskfcos_loss = gcv.loss.MaskFCOSLoss(from_logits=False, sparse_label=True, img_shape=(args.data_shape, args.data_shape),
                                          num_class=len(net.classes)+1, return_iou=False)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        tic = time.time()
        btic = time.time()
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            datas = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            ctr_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
            mask_targets = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
            matches = gluon.utils.split_and_load(batch[5], ctx_list=ctx, batch_axis=0)
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)

            with autograd.record():
                clsps, ctrps, boxps, maskps, maskcoeps = [], [], [], [], []
                for dat in datas:
                    cls_pred, ctr_pred, box_pred, masks, maskcoe_pred = net(dat)
                    clsps.append(cls_pred)
                    ctrps.append(ctr_pred)
                    boxps.append(box_pred)
                    maskps.append(masks)
                    maskcoeps.append(maskcoe_pred)
                sum_losses, cls_losses, ctr_losses, box_losses, mask_losses = \
                    maskfcos_loss(cls_targets, ctr_targets, box_targets, mask_targets, matches, clsps, ctrps, boxps, maskps, maskcoeps)
                autograd.backward(sum_losses)
            trainer.step(1, ignore_stale_grad=True) # normalize by batch_size
            if args.log_interval and not (i + 1) % args.log_interval:
                total_cls_loss, total_ctr_loss, total_box_loss, total_mask_loss = 0., 0., 0., 0.
                for cls_loss, ctr_loss, box_loss, mask_loss in zip(cls_losses, ctr_losses, box_losses, mask_losses):
                    total_cls_loss += cls_loss.asscalar()
                    total_ctr_loss += ctr_loss.asscalar()
                    total_box_loss += box_loss.asscalar()
                    total_mask_loss += mask_loss.asscalar()
                total_cls_loss /= batch_size
                total_ctr_loss /= batch_size
                total_box_loss /= batch_size
                total_mask_loss /= len(ctx)
                print_loss = {'cls_loss': total_cls_loss, 'ctr_loss': total_ctr_loss, \
                        'box_loss': total_box_loss, 'mask_loss': total_mask_loss}
                msg = ', '.join(['{}={:.3f}'.format(k, v) for k, v in print_loss.items()])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'\
                        .format(epoch, i, args.log_interval * batch_size / (time.time() \
                        - btic), msg))
                btic = time.time()

        logger.info('[Epoch {}] Training cost: {:.3f}'.format(
            epoch, (time.time() - tic)))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(1100)

    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    kwargs = {}
    net_name = "_".join(("maskfcos", args.network, str(args.data_shape), args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True, **kwargs)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
            net, train_dataset, val_dataset, MaskFCOSDefaultTrainTransform,
            MaskFCOSDefaultValTransform, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
