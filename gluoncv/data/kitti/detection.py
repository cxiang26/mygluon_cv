
import os
import numpy as np
from . import utils
from ..base import VisionDataset

__all__ = ['KITTIDetection']

class KITTIDetection(VisionDataset):
    """MS COCO detection dataset.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of str, default ['instances_val2017']
        Json annotations name.
        Candidates can be: instances_val2017, instances_train2017.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    min_object_area : float
        Minimum accepted ground-truth area, if an object's area is smaller than this value,
        it will be ignored.
    skip_empty : bool, default is True
        Whether skip images with no valid object. This should be `True` in training, otherwise
        it will cause undefined behavior.
    use_crowd : bool, default is True
        Whether use boxes labeled as crowd instance.

    """
    # CLASSES = ['Car', 'Truck', 'Van', 'Tram', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Misc']
    CLASSES = ['Car', 'Pedestrian', 'Cyclist']

    def __init__(self, root=os.path.join('d:/', 'Kitti/object'),
                 splits='training', transform=None, index_map=None,
                 min_object_area=0, skip_empty=True, use_crowd=True):
        super(KITTIDetection, self).__init__(root)
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._min_object_area = min_object_area
        self._skip_empty = skip_empty
        self._use_crowd = use_crowd
        self._splits = splits
        self._items = self._load_items()
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))

        if self._splits == 'training':
            self.num_samples = 3712
        elif self._splits == 'testing':
            self.num_samples = 3769
        else:
            print('Unknown split: %s' % (self._splits))
            exit(-1)

        self.image_dir = os.path.join(self._root, self._splits, 'image_2')
        self.calib_dir = os.path.join(self._root, self._splits, 'calib')
        self.label_dir = os.path.join(self._root, 'training', 'label_2')
        self.lidar_dir = os.path.join(self._root, self._splits, 'velodyne')
        # self.pred_dir = os.path.join(self._root, self._splits, 'pred')

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    def _load_items(self):
        ids = []
        if self._splits == 'training':
            lf = os.path.join(self._root, 'train'+'.txt')
        elif self._splits == 'testing':
            lf = os.path.join(self._root, 'val' + '.txt')
        else:
            raise TypeError('Getting error type splits {}'.format(self._splits))
        with open(lf, 'r') as f:
            ids += [line.strip() for line in f.readlines()]
        return ids

    def _get_image(self, idx):
        assert (idx < self.num_samples)
        name = self._items[idx]
        img_filename = os.path.join(self.image_dir, name+'.png')
        return utils.load_image(img_filename)

    def _get_lidar(self, idx):
        assert (idx < self.num_samples)
        name = self._items[idx]
        lidar_filename = os.path.join(self.lidar_dir, name+'.bin')
        return utils.load_velo_scan(lidar_filename)

    def _get_calibration(self, idx):
        assert (idx < self.num_samples)
        name = self._items[idx]
        calib_filename = os.path.join(self.calib_dir, name + '.txt')
        return utils.Calibration(calib_filename)

    def _get_label_objects(self, idx):
        assert (idx < self.num_samples)
        name = self._items[idx]
        label_filename = os.path.join(self.label_dir, name + '.txt')
        return utils.read_label(label_filename)

    def _get_label2D(self, idx):
        labels = self._get_label_objects(idx)
        label = []
        for la in labels:
            if la.type in ['Car', 'Truck', 'Van', 'Tram']:
                la.type = 'Car'
            elif la.type in ['Pedestrian', 'Person_sitting']:
                la.type = 'Pedestrian'
            elif la.type in ['Cyclist']:
                la.type = 'Cyclist'
            else:
                continue
            diffcult = int(la.occlusion >= 2)
            cls_id = self.index_map[la.type]
            label.append([la.xmin, la.ymin, la.xmax, la.ymax, cls_id, diffcult])
        return np.array(label)

    def _get_depth_map(self, idx):
        pass

    def _get_top_down(self, idx):
        pass

    def _isexist_pre_object(self, idx):
        assert (idx < self.num_samples and self._splits == 'training')
        pred_filename = os.path.join(self.pred_dir, '%06d.txt' % (idx))
        return os.path.exists(pred_filename)


    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self._get_image(idx)
        #lidar = self._get_lidar(idx)
        #cal = self._get_calibration(idx)
        label2d = self._get_label2D(idx)
        # pred = self._get_pred_objects(idx)

        if self._transform is not None:
            return self._transform(img, label2d)
        return img, label2d

# def draw_projected_box3d(image, qs, color=(0,255,0), thickness=2):
#     ''' Draw 3d bounding box in image
#         qs: (8,3) array of vertices for the 3d box in following order:
#             1 -------- 0
#            /|         /|
#           2 -------- 3 .
#           | |        | |
#           . 5 -------- 4
#           |/         |/
#           6 -------- 7
#     '''
#     qs = qs.astype(np.int32)
#     for k in range(0,4):
#        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
#        i,j=k,(k+1)%4
#        # use LINE_AA for opencv3
#        #cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
#        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
#        i,j=k+4,(k+1)%4 + 4
#        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
#
#        i,j=k,k+4
#        cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
#     return image

# if __name__ == '__main__':
#     import cv2
#     data = KITTIDetection()
#     img, label, lidar, cal = data.__getitem__(1000)
#     print(label)
#     for la in label:
#         if la.type == 'DontCare':
#             continue
#         else:
#             box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(la, cal.P)
#             img = draw_projected_box3d(img, box3d_pts_2d)
#     cv2.namedWindow('results')
#     cv2.imshow('results',img)
#     cv2.waitKey()
#     print(img.dtype, label.dtype)
