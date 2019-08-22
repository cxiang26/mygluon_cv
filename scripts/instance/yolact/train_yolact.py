"""Train SSD"""
import argparse
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad, Append
from gluoncv.data.transforms.presets.yolact import YOLACTDefaultTrainTransform
from gluoncv.data.transforms.presets.yolact import YOLACTDefaultValTransform
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.data import batchify
from gluoncv.utils.metrics.coco_instance import COCOInstanceMetric
from gluoncv.utils import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLACT networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=550,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default='8',
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='2',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=55,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='20,40,47, 51',
                        help='epochs at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='/media/HDD_4TB/xcq/experiments/yolact/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=10,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    # FPN options
    parser.add_argument('--use-fpn', action='store_true', default=True,
                        help='Whether to use feature pyramid network.')
    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        train_dataset = gdata.COCOInstance(root='/media/SSD_1TB/coco/', splits='instances_train2017')
        val_dataset = gdata.COCOInstance(root='/media/SSD_1TB/coco/', splits='instances_val2017', skip_empty=False)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    scale = 4
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors, _, _ = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets, masks, matches
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLACTDefaultTrainTransform(width, height, anchors, scale=scale)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLACTDefaultValTransform(width, height, scale=4)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


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
        _h = (_h >= x1.expand_dims(axis=-1)) * (_h <= x2.expand_dims(axis=-1))
        _w = (_w >= y1.expand_dims(axis=-1)) * (_w <= y2.expand_dims(axis=-1))
        _mask = mx.nd.batch_dot(_h.expand_dims(axis=-1), _w.expand_dims(axis=-1), transpose_b=True)
    masks = _mask * masks
    return masks

def global_aware(masks):
    _, h, w = masks.shape
    masks = masks.reshape((0, -1))
    masks = masks - mx.nd.mean(masks, axis=-1, keepdims=True)
    std = mx.nd.sqrt(mx.nd.mean(mx.nd.square(masks), axis=-1, keepdims=True))
    masks = (masks / (std + 1e-6)).reshape((0, h, w))
    return masks

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    # clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    # if not args.disable_hybridization:
    #     net.hybridize(static_alloc=args.static_alloc)
    net.hybridize()
    for ib, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        det_info = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        for x, det_inf in zip(data, det_info):
            det_id, det_score, det_bbox, det_maskeoc, det_mask = net(x)
            for i in range(det_bbox.shape[0]):
                # numpy everything
                det_bbox_t = det_bbox[i] # det_bbox_t: [x1, y1, x2, y2]
                det_id_t = det_id[i].asnumpy()
                det_score_t = det_score[i].asnumpy()
                det_maskeoc_t = det_maskeoc[i]
                det_mask_t = det_mask[i]
                full_mask = mx.nd.dot(det_maskeoc_t, det_mask_t)
                im_height, im_width, h_scale, w_scale = det_inf[i].asnumpy()
                im_height, im_width = int(round(im_height / h_scale)), int(
                    round(im_width / w_scale))
                full_mask = global_aware(full_mask)
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
                assert det_bbox_t.shape[0] == det_id_t.shape[0] == det_score_t.shape[0] == full_masks.shape[0], \
                    print(det_bbox_t.shape[0], det_id_t.shape[0], det_score_t.shape[0], full_masks.shape[0])
                eval_metric.update(det_bbox_t, det_id_t, det_score_t, full_masks)
    return eval_metric.get()

def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])

    # mbox_loss = gcv.loss.SSDMultiBoxLoss()
    mbox_loss = gcv.loss.YOLACTMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')
    sq_metric = mx.metric.Loss('SigmoidBCE')

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
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        ce_metric.reset()
        smoothl1_metric.reset()
        sq_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
            mask_targets = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
            matches = gluon.utils.split_and_load(batch[4], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                cls_preds = []
                box_preds = []
                masks = []
                maskeocs = []
                bts = []
                for x, bt in zip(data, box_targets):
                    cls_pred, box_pred, anchor, maskeoc, mask = net(x)
                    bts.append(net.bbox_decoder(bt, anchor))
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                    masks.append(mask)
                    maskeocs.append(maskeoc)
                sum_loss, cls_loss, box_loss, mask_loss = mbox_loss(
                    cls_preds, box_preds, masks, maskeocs, cls_targets, box_targets, mask_targets, matches, bts)


                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            sq_metric.update(0, [l * batch_size for l in mask_loss])
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                name3, loss3 = sq_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f},'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3))
            btic = time.time()
            break

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        name3, loss3 = sq_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0) or (epoch >= 50):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)

if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    if args.use_fpn:
        net_name = '_'.join(('yolact', str(args.data_shape), 'fpn', args.network, args.dataset))
    else:
        net_name = '_'.join(('yolact', str(args.data_shape), args.network, args.dataset))
    args.save_prefix += net_name
    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
    else:
        net = get_model(net_name, pretrained_base=True)
        async_net = net
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()



    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    train_data, val_data = get_dataloader(
        async_net, train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
