# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys

import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
# from IPython import embed

class FCOSTargetGenerator(nn.Block):
    """Generate FCOS targets"""
    def __init__(self, retina_stages=5, base_stride=128,
                 valid_range=[(384, np.inf), (192, 384), (96, 192), (48, 96), (0, 48)], **kwargs):

        super(FCOSTargetGenerator, self).__init__(**kwargs)
        self._stages = retina_stages
        self._stride = base_stride
        self._valid_range = valid_range

    def generate_targets(self, img, boxes):
        """
        img : [H, W, 3]
        boxes : [N, 5]
        """
        rh, rw, _ = img.shape
        rx = nd.arange(0, rw).reshape((1, -1))
        ry = nd.arange(0, rh).reshape((-1, 1))
        sx = nd.tile(rx, reps=(rh, 1))
        sy = nd.tile(ry, reps=(1, rw))

        x0, y0, x1, y1, _ = nd.split(boxes, 5, axis=-1, squeeze_axis=True)
        areas = (x1 - x0) * (y1 - y0)
        # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        boxes_id = nd.argsort(areas)
        boxes_id = nd.concat(nd.array([-1]), boxes_id, dim=0)

        boxes = nd.take(boxes, nd.argsort(areas)) # min -> max
        boxes = nd.concat(nd.zeros((1, 5)), boxes, dim=0) # for gt assign confusion
        x0, y0, x1, y1, cls = nd.split(boxes, num_outputs=5, axis=-1, squeeze_axis=True)
        n = boxes.shape[0]

        # [H, W, N]
        of_l = sx.reshape(-2, 1) - nd.expand_dims(nd.expand_dims(x0, axis=0), axis=0)
        of_t = sy.reshape(-2, 1) - nd.expand_dims(nd.expand_dims(y0, axis=0), axis=0)
        of_r = -(sx.reshape(-2, 1) - nd.expand_dims(nd.expand_dims(x1, axis=0), axis=0))
        of_b = -(sy.reshape(-2, 1) - nd.expand_dims(nd.expand_dims(y1, axis=0), axis=0))

        # [H, W, N]
        eps = 1e-5
        # ctr = nd.minimum(of_l, of_r) / (nd.maximum(of_l, of_r) + eps) * \
        #         nd.minimum(of_t, of_b) / (nd.maximum(of_t, of_b) + eps)
        # ctr = nd.sqrt(nd.abs(ctr))
        # ctr[:, :, 0] = 0

        # # flat ctr
        of_l = of_l * (of_l > 0)
        of_r = of_r * (of_r > 0)
        of_t = of_t * (of_t > 0)
        of_b = of_b * (of_b > 0)
        # ctr2 = nd.minimum(of_l, of_r) / (nd.maximum(of_l, of_r) + of_l + of_r) * \
        #         nd.minimum(of_t, of_b) / (nd.maximum(of_t, of_b) + of_t + of_b)
        # ctr2 = 3 * nd.sqrt(nd.abs(ctr2))
        # ctr2[:, :, 0] = 0

        # slim ctr
        # ctr = nd.minimum(of_l, of_r) / (nd.maximum(of_l, of_r) + nd.abs(of_l - of_r) + eps) * \
        #        nd.minimum(of_t, of_b) / (nd.maximum(of_t, of_b) + nd.abs(of_t - of_b) + eps)
        ctr = nd.minimum(of_l, of_r) / (nd.maximum(of_l, of_r) + eps) * \
                nd.minimum(of_t, of_b) / (nd.maximum(of_t, of_b) + eps)
        # ctr = nd.power(0.8, 0.1 * nd.sqrt(nd.square(of_l - of_r) + nd.square(of_t - of_b) + eps))
        # ctr = nd.power(0.8, nd.sqrt(nd.abs(of_l - of_r) + nd.abs(of_t - of_b) + eps))
        ctr = nd.sqrt(nd.abs(ctr))
        ctr[:, :, 0] = 0
        # [H, W, N, 4]
        offsets = nd.concat(of_l.reshape(-2, 1), of_t.reshape(-2, 1),
                            of_r.reshape(-2, 1), of_b.reshape(-2, 1), dim=-1)

        # fh = int(np.ceil(((rh + 1) / 2) // 2 / 2))
        # fw = int(np.ceil(((rw + 1) / 2) // 2 / 2))
        fh = int(np.ceil(np.ceil(np.ceil(rh / 2) / 2) / 2))
        fw = int(np.ceil(np.ceil(np.ceil(rw / 2) / 2) / 2))

        fm_list = []
        for i in range(self._stages):
            fm_list.append((fh, fw))
            fh = int(np.ceil(fh / 2))
            fw = int(np.ceil(fw / 2))
        fm_list = fm_list[::-1]
        cls_targets = []
        ctr_targets = []
        box_targets = []
        match_targets = []
        stride = self._stride
        for i in range(self._stages):
            fh, fw = fm_list[i]
            # cls_target = nd.zeros((fh, fw))
            # box_target = nd.zeros((fh, fw, 4))
            # ctr_target = nd.zeros((fh, fw))
            # match_target = nd.zeros((fh, fw))

            cx = nd.arange(0, fw).reshape((1, -1))
            cy = nd.arange(0, fh).reshape((-1, 1))
            sx = nd.tile(cx, reps=(fh, 1))
            sy = nd.tile(cy, reps=(1, fw))
            syx = nd.stack(sy.reshape(-1), sx.reshape(-1)).transpose().astype('int32')
            # bugs in this type
            # bx = sxy[:, 0] * stride + nd.floor(sxy[:, 0] / 2).astype(np.int32)
            # by = sxy[:, 1] * stride + nd.floor(sxy[:, 1] / 2).astype(np.int32)
            by, bx = nd.split(syx*stride, 2, axis=-1, squeeze_axis=True)
            # by = syx[:, 0] * stride
            # bx = syx[:, 1] * stride

            # [FH*FW, N, 4]
            of_byx = nd.take(offsets.reshape((-1, n, 4)), by*740+bx)
            of_ctr = nd.take(ctr.reshape((-1, n)), by*740 + bx)
            # of_byx = offsets[by, bx]
            # ctr_aware = ctr[by, bx]
            # of_byx = nd.gather_nd(offsets, indices=byx.transpose())
            min_vr, max_vr = self._valid_range[i]
            # [FH*FW, N]
            is_in_box = nd.prod(of_byx > 0, axis=-1)
            is_valid_area = (of_byx.max(axis=-1) >= min_vr) * (of_byx.max(axis=-1) <= max_vr)
            # [FH*FW, N]
            valid_pos = nd.elemwise_mul(is_in_box, is_valid_area) * of_ctr
            # valid_pos = nd.elemwise_mul(is_in_box, is_valid_area)
            # of_valid = nd.zeros((fh, fw, n))
            # of_valid[syx[:, 0], syx[:, 1], :] = valid_pos * ctr_aware # 1, 0
            of_valid = valid_pos.reshape((fh, fw, n))
            of_valid[:, :, 0] = 0
            # [FH, FW]
            # gt_inds = nd.argmax(of_valid, axis=-1)
            gt_inds = nd.argmax(of_valid, axis=-1).reshape(-1)
            # box targets
            box_target = nd.take(boxes, gt_inds).slice_axis(begin=0, end=4, axis=-1)
            # box_target[syx[:, 0], syx[:, 1]] = boxes[gt_inds[syx[:, 0], syx[:, 1]], :4]
            # box_target = box_target.reshape(-1, 4)

            # cls targets
            cls_target = nd.take(cls, gt_inds)
            # cls_target[syx[:, 0], syx[:, 1]] = cls[gt_inds[syx[:, 0], syx[:, 1]]]
            # cls_target = cls_target.reshape(-1)

            # match targets the number of matches less than ctr targets
            match_gt_inds = nd.argmax(of_valid * (of_valid > 0.01), axis=-1).reshape(-1)
            match_target = nd.take(boxes_id, match_gt_inds)
            # match_target[syx[:, 0], syx[:, 1]] = boxes_id[match_gt_inds[syx[:,0], syx[:,1]]]
            # match_target = match_target.reshape(-1)

            # ctr targets
            ctr_target = nd.pick(of_ctr, gt_inds)
            # ctr_target[syx[:, 0], syx[:, 1]] = ctr[by, bx, gt_inds[syx[:, 0], syx[:, 1]]]
            # ctr_target = ctr_target.reshape(-1)
            box_targets.append(box_target)
            cls_targets.append(cls_target)
            ctr_targets.append(ctr_target)
            stride = int(stride / 2)
            match_targets.append(match_target)
        box_targets = nd.concat(*box_targets, dim=0)
        cls_targets = nd.concat(*cls_targets, dim=0)
        ctr_targets = nd.concat(*ctr_targets, dim=0)
        match_targets = nd.concat(*match_targets, dim=0)
        return cls_targets, ctr_targets, box_targets, match_targets

    def forward(self, img, boxes):
        pass

class FCOSBoxConverter(nn.HybridBlock):
    """This function is used to convert box_preds(l,t,r,b)
       to corner(x1, y1, x2, y2) format, which then used
       to compute IoULoss.
    """
    def __init__(self, **kwargs):
        super(FCOSBoxConverter, self).__init__(**kwargs)
        pass

    def hybrid_forward(self, F, box_preds, box_cords):
        """
        box_preds : [B, N, 4]
        box_cords : [B, N, 2]
            coordinates for the feature map corresponding to original image.
        """
        cx, cy = F.split(box_cords, num_outputs=2, axis=-1)
        pl, pt, pr, pb = F.split(box_preds, num_outputs=4, axis=-1)
        x1 = F.broadcast_minus(cx, pl)
        y1 = F.broadcast_minus(cy, pt)
        x2 = F.broadcast_add(cx, pr)
        y2 = F.broadcast_add(cy, pb)
        boxes = F.concat(x1, y1, x2, y2, dim=2)
        return boxes

