"""Transforms for RCNN series."""
from __future__ import absolute_import

import copy
from random import randint

import numpy as np
import mxnet as mx
from mxnet import nd

from .. import bbox as tbbox
from .. import image as timage
from .. import mask as tmask
from .. import experimental

__all__ = ['transform_test', 'load_test',
           'FCOSDefaultTrainTransform', 'FCOSDefaultValTransform',
           'MaskFCOSDefaultTrainTransform', 'MaskFCOSDefaultValTransform']

def transform_test(imgs, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [mx.image.imread(f) for f in filenames]
    return transform_test(imgs, short, max_size, mean, std)


class FCOSDefaultTrainTransform(object):
    def __init__(self, short=600, max_size=1000, base_stride=128,
                 valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 flip_p=0.5, retina_stages=5, num_class=80, **kwargs):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._random_resize = isinstance(self._short, (tuple, list))
        self._flip_p = flip_p
        self._num_class = num_class

        from ....model_zoo.fcos.fcos_target import FCOSTargetGenerator
        self._target_generator = FCOSTargetGenerator(retina_stages=retina_stages,
                                    base_stride=base_stride, valid_range=valid_range)

    def __call__(self, src, label):
        "Apply transform to training image/label."
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        if self._random_resize:
            short = randint(self._short(0), self._short[1])
        else:
            short = self._short
        img = timage.resize_short_within(src, short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=self._flip_p)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # generate training targets for fcos
        tbox = mx.nd.round(mx.nd.array(bbox))
        tbox[:, 4] += 1. #
        cls_targets, ctr_targets, box_targets, cor_targets = \
            self._target_generator.generate_targets(img, tbox)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        return img, cls_targets, ctr_targets, box_targets, cor_targets


class FCOSDefaultValTransform(object):
    def __init__(self, short=600, max_size=1000, base_stride=128, retina_stages=5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size
        self._base_stride = base_stride
        self._retina_stages = retina_stages

    def _generate_coordinates(self, img):
        h, w, _ = img.shape
        fh = int(np.ceil(np.ceil(np.ceil(h / 2) / 2) / 2))
        fw = int(np.ceil(np.ceil(np.ceil(w / 2) / 2) / 2))
        stride = self._base_stride
        #
        fm_list = []
        for i in range(self._retina_stages):
            fm_list.append((fh, fw))
            fh = int(np.ceil(fh / 2))
            fw = int(np.ceil(fw / 2))
        fm_list = fm_list[::-1]
        #
        cor_targets = []
        for i in range(self._retina_stages):
            fh, fw = fm_list[i]
            cx = nd.arange(0, fw).reshape((1, -1))
            cy = nd.arange(0, fh).reshape((-1, 1))
            sx = nd.tile(cx, reps=(fh, 1))
            sy = nd.tile(cy, reps=(1, fw))
            syx = nd.stack(sy.reshape(-1), sx.reshape(-1)).transpose()
            by = syx[:, 0] * stride
            bx = syx[:, 1] * stride
            cor_targets.append(nd.stack(bx, by, axis=1))
            stride = int(stride / 2)
        cor_targets = nd.concat(*cor_targets, dim=0)
        return cor_targets

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))
        im_scale = h / float(img.shape[0])

        # generate coords back to ori image
        cor_targets = self._generate_coordinates(img)

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32'), cor_targets, mx.nd.array([im_scale])

class MaskFCOSDefaultTrainTransform(object):
    def __init__(self, short=600, base_stride=128,
                 valid_range=[(512, np.inf), (256, 512), (128, 256), (64, 128), (0, 64)],
                 scale=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 flip_p=0.5, retina_stages=5, num_class=80):
        self._width = short
        self._height = short
        self._scale = scale
        self._mean = mean
        self._std = std
        self._flip_p = flip_p
        self._num_class = num_class

        from ....model_zoo.mask_fcos.fcos_target import FCOSTargetGenerator
        self._target_generator = FCOSTargetGenerator(retina_stages=retina_stages,
                                    base_stride=base_stride, valid_range=valid_range)

    def __call__(self, src, label, segm):
        "Apply transform to training image/label."
        # resize shorter side but keep in max_size
        if np.random.uniform(0, 1) > 0.5:
            img = experimental.image.random_color_distort(src)
            bbox = label
        else:
            img, bbox = src, label

        # random expansion with prob 0.5
        # if np.random.uniform(0, 1) > 0.5:
        #     img, expand = timage.random_expand(img, max_ratio=1.5, fill=[m * 255 for m in self._mean])
        #     bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        #     segm = [tmask.expand(polys, x_offset=expand[0], y_offset=expand[1]) for polys in segm]
        # else:
        #     img, bbox, segm = img, label, segm

        # random cropping
        # h, w, _ = img.shape
        # bbox, crop = experimental.bbox.random_crop_with_constraints(label, (w, h), min_scale=0.8)
        # if bbox.shape[0] == label.shape[0]:
        #     x0, y0, w, h = crop
        #     img = mx.image.fixed_crop(img, x0, y0, w, h)
        #     segm = [tmask.crop(polys, x0, y0, w, h) for polys in segm]
        # else:
        #     bbox = label

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
        segm = [tmask.flip(polys, size=(w, h), flip_x=flips[0]) for polys in segm]

        # resize with random interpolation
        h, w, _ = img.shape
        masks_width, masks_height = int(np.ceil(self._width / (self._scale*2))*2), \
                                    int(np.ceil(self._height / (self._scale*2))*2)
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        segm = [tmask.resize(polys, in_size=(w, h), out_size=(masks_width, masks_height)) for polys in segm]

        masks = [mx.nd.array(tmask.to_mask(polys, (masks_width, masks_height))) for polys in segm]
        masks = mx.nd.stack(*masks, axis=0)

        assert masks.shape[0] == bbox.shape[0],\
            print('the number of masks is:', masks.shape[0], 'but the number of bbox is:', bbox.shape[0])

        # generate training targets for fcos
        tbox = mx.nd.round(mx.nd.array(bbox))
        tbox[:, 4] += 1. #
        cls_targets, ctr_targets, box_targets, matches = \
            self._target_generator.generate_targets(img, tbox)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        gt_masks = mx.nd.zeros(shape=(100, masks_width, masks_height))
        assert masks.shape[0] <= 100, print(masks.shape[0])
        gt_masks[:masks.shape[0], :, :] = masks

        return img, cls_targets, ctr_targets, box_targets, gt_masks, matches


class MaskFCOSDefaultValTransform(object):
    def __init__(self, short=800, base_stride=128, retina_stages=5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._width = short
        self._height = short
        # self._max_size = max_size
        self._base_stride = base_stride
        self._retina_stages = retina_stages

    def __call__(self, src, label, segm):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        h_scale, w_scale = float(img.shape[0] / h), float(img.shape[1] / w)

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img,  mx.nd.array([img.shape[-2], img.shape[-1], h_scale, w_scale])
