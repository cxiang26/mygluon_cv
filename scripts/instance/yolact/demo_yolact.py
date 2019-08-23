"""Mask RCNN Demo script."""
import os
import argparse
import mxnet as mx
import gluoncv as gcv
from gluoncv.data.transforms import presets
from matplotlib import pyplot as plt
import time
def parse_args():
    parser = argparse.ArgumentParser(description='Test with Mask RCNN networks.')
    parser.add_argument('--network', type=str, default='yolact_550_fpn_resnet50_v1b_coco',
                        help="Mask RCNN full network name")
    parser.add_argument('--images', type=str, default='',
                        help='Test images, use comma to split multiple.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Testing with GPUs, you can specify 0 for example.')
    parser.add_argument('--pretrained', type=str, default='/mnt/mdisk/xcq/results/yolact/yolact_550_fpn_resnet50_v1b_coco_best.params',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    args = parser.parse_args()
    return args

def crop(bboxes, masks):
    scale = 4
    h, w = masks.shape[1], masks.shape[2]
    b = masks.shape[0]
    ctx = masks.context
    _h = mx.nd.arange(h, ctx = ctx)
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

if __name__ == '__main__':
    args = parse_args()
    # context list
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = [mx.cpu()] if not ctx else ctx

    # grab some image if not specified
    if not args.images.strip():
        gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                           'gluoncv/detection/biking.jpg?raw=true', 'biking.jpg')
        image_list = ['biking.jpg']
    else:
        image_list = [x.strip() for x in args.images.split(',') if x.strip()]

    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(args.network, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(args.network, pretrained=False)
        net.load_parameters(args.pretrained)
    net.collect_params().reset_ctx(ctx)

    for image in image_list:
        x, img = presets.yolact.load_test(image, short=550)
        x = x.as_in_context(ctx[0])
        mx.nd.waitall()
        tic = time.time()
        ids, scores, bboxes, maskeoc, masks = net(x)
        mx.nd.waitall()
        print(time.time()-tic)
        masks = mx.nd.dot(maskeoc.squeeze(), masks.squeeze())
        tic = time.time()
        masks = crop(bboxes.squeeze(), masks).asnumpy()
        print(time.time()-tic)
        ids = ids.squeeze().asnumpy()
        scores = scores.squeeze().asnumpy()
        bboxes = bboxes.squeeze().asnumpy()
        bboxes[:, (0, 2)] = bboxes[:, (0, 2)] / (x.shape[3] / img.shape[1])
        bboxes[:, (1, 3)] = bboxes[:, (1, 3)] / (x.shape[2] / img.shape[0])

        new_masks = gcv.utils.viz.expand_yolactmask(masks, bboxes, (img.shape[1], img.shape[0]), scores)
        img = gcv.utils.viz.plot_mask(img, new_masks, alpha=0.3)

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(1, 1, 1)
        ax = gcv.utils.viz.plot_bbox(img, bboxes, scores, ids,
                                     class_names=net.classes, ax=ax)
        plt.show()
