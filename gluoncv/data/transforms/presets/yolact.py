"""Transforms described in https://arxiv.org/abs/1512.02325."""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import mask as tmask
from .. import experimental

__all__ = ['transform_test', 'load_test', 'YOLACTDefaultTrainTransform', 'YOLACTDefaultValTransform']

def transform_test(imgs, short, max_size=1024, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
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

def load_test(filenames, short, max_size=1024, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or iterable of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
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


class YOLACTDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    iou_thresh : float
        IOU overlap threshold for maximum matching, default is 0.5.
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, width, height, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2), scale=8,
                 **kwargs):
        self._width = width
        self._height = height
        self._anchors = anchors
        self._mean = mean
        self._std = std
        self._scale = scale
        if anchors is None:
            return

        # since we do not have predictions yet, so we ignore sampling here
        from ....model_zoo.yolact.target import YOLACTTargetGenerator
        self._target_generator = YOLACTTargetGenerator(
            iou_thresh=iou_thresh, stds=box_norm, negative_mining_ratio=-1, **kwargs)

    def __call__(self, src, label, segm):
        """Apply transform to training image/label."""
        # random color jittering
        img = experimental.image.random_color_distort(src)
        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, max_ratio=2, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
            segm = [tmask.expand(polys, x_offset=expand[0], y_offset=expand[1]) for polys in segm]
        else:
            img, bbox, segm = img, label, segm

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)
        segm = [tmask.crop(polys, x0, y0, w, h) for polys in segm]

        # resize with random interpolation
        h, w, _ = img.shape
        masks_width, masks_height = int(np.ceil(self._width/self._scale)), int(np.ceil(self._height/self._scale))
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        segm = [tmask.resize(polys, in_size=(w, h), out_size=(masks_width, masks_height)) for polys in segm]

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
        segm = [tmask.flip(polys, size=(masks_width, masks_height), flip_x=flips[0]) for polys in segm]

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        masks = [mx.nd.array(tmask.to_mask(polys, (masks_width, masks_height))) for polys in segm]
        masks = mx.nd.stack(*masks, axis=0)

        if self._anchors is None:
            return img, bbox.astype(img.dtype), masks

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_masks = mx.nd.zeros(shape=(100, masks_width, masks_height))
        assert masks.shape[0] <= 100, print(masks.shape[0])
        gt_masks[:masks.shape[0], :, :] = masks
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        cls_targets, box_targets, matches = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0], gt_masks, matches.squeeze()


# class YOLACTDefaultValTransform(object):
    # """Default SSD validation transform.
    #
    # Parameters
    # ----------
    # width : int
    #     Image width.
    # height : int
    #     Image height.
    # mean : array-like of size 3
    #     Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    # std : array-like of size 3
    #     Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    #
    # """
    # def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale=8):
    #     self._width = width
    #     self._height = height
    #     self._mean = mean
    #     self._std = std
    #     self._scale = scale
    #
    # def __call__(self, src, label, segm):
    #     """Apply transform to validation image/label."""
    #     # resize
    #     h, w, _ = src.shape
    #     mask_width, mask_height = int(self._width/self._scale), int(self._height/self._scale)
    #     img = timage.imresize(src, self._width, self._height, interp=9)
    #     bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))
    #     segm = [tmask.resize(polys, in_size=(w, h), out_size=(mask_width, mask_height)) for polys in segm]
    #
    #     img = mx.nd.image.to_tensor(img)
    #     img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
    #     masks = [mx.nd.array(tmask.to_mask(polys, (mask_width, mask_height))) for polys in segm]
    #     masks = mx.nd.stack(*masks, axis=0)
    #     gt_masks = mx.nd.zeros(shape=(25, mask_width, mask_height))
    #     assert masks.shape[0] <= 25, "gt masks has less channels!"
    #     gt_masks[:masks.shape[0], :, :] = masks
    #     return img, bbox.astype(img.dtype), gt_masks

class YOLACTDefaultValTransform(object):

    def __init__(self, width, height,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), scale=8):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std
        self._scale = scale

    def __call__(self, src, label, segm):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        # mask_width, mask_height = int(self._width/self._scale), int(self._height/self._scale)
        img = timage.imresize(src, self._width, self._height, interp=9)
        # segm = [tmask.resize(polys, in_size=(w, h), out_size=(mask_width, mask_height)) for polys in segm]
        h_scale, w_scale = float(img.shape[0]/h), float(img.shape[1]/w)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        # masks = [mx.nd.array(tmask.to_mask(polys, (mask_width, mask_height))) for polys in segm]
        # masks = mx.nd.stack(*masks, axis=0)
        # gt_masks = mx.nd.zeros(shape=(25, mask_width, mask_height))
        # assert masks.shape[0] <= 25, "gt masks has less channels!"
        # gt_masks[:masks.shape[0], :, :] = masks
        return img, mx.nd.array([img.shape[-2], img.shape[-1], h_scale, w_scale])