import os
from tqdm import tqdm
import numpy as np
import argparse

import mxnet as mx
from mxnet import gluon, image, nd
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascal_aug',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxilary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxilary loss weight')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        True, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=1,
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default='/home/xcq/.mxnet/models/fcn_resnet101_ade-d544440a.params',
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default='fcn_resnet101_voc',
                        help='evaluating on model zoo model')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= True,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args

def test(args):
    # output folder
    means = nd.array([123, 117, 104])
    std = nd.array([58.395, 57.12, 57.375])
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # dataset and dataloader
    if args.eval:
        img = image.imread('./0000000152.png')
        img = img.astype('float32')
        img = img - means
        img = img / std
        img = nd.transpose(img, (2, 0, 1))
        img = nd.expand_dims(img, axis=0)
        testset = get_segmentation_dataset(
            args.dataset, split='val', mode='testval', transform=input_transform, root='/mnt/mdisk/xcq/VOCdevkit/')
        total_inter, total_union, total_correct, total_label = \
            np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    else:
        testset = get_segmentation_dataset(
            args.dataset, split='test', mode='test', transform=input_transform, root='/mnt/mdisk/xcq/VOCdevkit/')
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
                                       backbone=args.backbone, norm_layer=args.norm_layer,
                                       norm_kwargs=args.norm_kwargs, aux=args.aux,
                                       base_size=args.base_size, crop_size=args.crop_size)
        # load pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        if os.path.isfile(args.resume):
            model.load_parameters(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))
    print(model)
    predicts = model(img)
    evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    for i, (data, dsts) in enumerate(tbar):
        if args.eval:
            predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
            targets = [target.as_in_context(predicts[0].context) \
                       for target in dsts]
            metric.update(targets, predicts)
            pixAcc, mIoU = metric.get()
            tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            im_paths = dsts
            predicts = evaluator.parallel_forward(data)
            for predict, impath in zip(predicts, im_paths):
                predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 1)).asnumpy() + \
                    testset.pred_offset
                mask = get_color_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))

if __name__ == "__main__":
    args = parse_args()
    args.test_batch_size = args.ngpus
    print('Testing model: ', args.resume)
    test(args)
