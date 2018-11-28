
from __future__ import division, absolute_import, print_function
import numpy as np


__all__ = ['rotate_point_cloud', 'rotate_perturbation_point_cloud', 'random_point_dropout', 'random_scale_point_cloud',
           'rotate_point_cloud_by_angle', 'rotate_point_cloud_z', 'jitter_point_cloud', 'normalize_point_cloud']

# class Augmenter(object):
#     """Image Augmenter base class"""
#     def __init__(self, **kwargs):
#         self._kwargs = kwargs
#         for k, v in self._kwargs.items():
#             if isinstance(v, nd.NDArray):
#                 v = v.asnumpy()
#             if isinstance(v, np.ndarray):
#                 v = v.tolist()
#                 self._kwargs[k] = v
#
#     def dumps(self):
#         """Saves the Augmenter to string
#
#         Returns
#         -------
#         str
#             JSON formatted string that describes the Augmenter.
#         """
#         return json.dumps([self.__class__.__name__.lower(), self._kwargs])
#
#
#     def __call__(self, src):
#         """Abstract implementation body"""
#         raise NotImplementedError("Must override implementation.")
#
# class rotate_point_cloud(Augmenter):
#     """ Randomly rotate the point clouds to augument the dataset
#         rotation is per shape based along up direction
#         Input:
#           Nx3 array, original point clouds
#         Return:
#           Nx3 array, rotated point clouds
#     """
#     def __init__(self):
#         super(rotate_point_cloud, self).__init__()
#     def __call__(self, point_data):
#         rotated_data = np.zeros(point_data.shape, dtype=np.float32)
#         rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(rotation_angle)
#         sinval = np.sin(rotation_angle)
#         rotation_matrix = np.array([[cosval, 0, sinval],
#                                     [0, 1, 0],
#                                     [-sinval, 0, cosval]])
#         rotated_data = np.dot(point_data, rotation_matrix)
#         return rotated_data
#
# class rotate_point_cloud_z(Augmenter):
#     """ Randomly rotate the point clouds to augument the dataset
#         rotation is per shape based along up direction
#         Input:
#           Nx3 array, original point clouds
#         Return:
#           Nx3 array, rotated point clouds
#     """
#     def __init__(self):
#         super(rotate_point_cloud_z, self).__init__()
#     def __call__(self, point_data):
#         rotated_data = np.zeros(point_data.shape, dtype=np.float32)
#         rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(rotation_angle)
#         sinval = np.sin(rotation_angle)
#         rotation_matrix = np.array([[cosval, sinval, 0],
#                                     [-sinval, cosval, 0],
#                                     [0, 0, 1]])
#         rotated_data = np.dot(point_data, rotation_matrix)
#         return rotated_data
#
# class rotate_point_cloud_by_angle(Augmenter):
#     """ Rotate the point cloud along up direction with certain angle.
#         Input:
#           Nx3 array, original point clouds
#         Return:
#           Nx3 array, rotated  point clouds
#     """
#     def __init__(self, rotation_angle):
#         super(rotate_point_cloud_by_angle, self).__init__()
#         self.rotation_angle = rotation_angle
#     def __call__(self, point_data):
#         rotated_data = np.zeros(point_data.shape, dtype=np.float32)
#         #rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(self.rotation_angle)
#         sinval = np.sin(self.rotation_angle)
#         rotation_matrix = np.array([[cosval, 0, sinval],
#                                     [0, 1, 0],
#                                     [-sinval, 0, cosval]])
#         rotated_data = np.dot(point_data, rotation_matrix)
#         return rotated_data
#
# class rotate_perturbation_point_cloud(Augmenter):
#     """ Randomly perturb the point clouds by small rotations
#         Input:
#           Nx3 array, original point clouds
#         Return:
#           Nx3 array, rotated point clouds
#     """
#     def __init__(self, angle_sigma=0.06, angle_clip=0.18):
#         super(rotate_perturbation_point_cloud, self).__init__()
#         self.angle_sigma = angle_sigma
#         self.angle_clip = angle_clip
#     def __call__(self, point_data):
#         rotated_data = np.zeros(point_data.shape, dtype=np.float32)
#         angles = np.clip(self.angle_sigma*np.random.randn(3), -self.angle_clip, self.angle_clip)
#         Rx = np.array([[1,0,0],
#                        [0,np.cos(angles[0]),-np.sin(angles[0])],
#                        [0,np.sin(angles[0]),np.cos(angles[0])]])
#         Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
#                        [0,1,0],
#                        [-np.sin(angles[1]),0,np.cos(angles[1])]])
#         Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
#                        [np.sin(angles[2]),np.cos(angles[2]),0],
#                        [0,0,1]])
#         R = np.dot(Rz, np.dot(Ry,Rx))
#         rotated_data = np.dot(point_data, R)
#         return rotated_data
#
#
# class jitter_point_cloud(Augmenter):
#     """ Randomly jitter points. jittering is per point.
#         Input:
#           Nx3 array, original point clouds
#         Return:
#           Nx3 array, jittered point clouds
#     """
#     def __init__(self,sigma=0.01, clip=0.05):
#         super(jitter_point_cloud, self).__init__()
#         self.sigma = sigma
#         self.clip = clip
#     def __call__(self, point_data):
#         N, C = point_data.shape
#         assert(self.clip > 0)
#         jittered_data = np.clip(self.sigma * np.random.randn(N, C), -1*self.clip, self.clip)
#         jittered_data += point_data
#         return jittered_data
#
# class shift_point_cloud(Augmenter):
#     """ Randomly shift point cloud. Shift is per point cloud.
#         Input:
#           Nx3 array, original point clouds
#         Return:
#           Nx3 array, shifted point clouds
#     """
#     def __init__(self, shift_range=0.1):
#         super(shift_point_cloud, self).__init__()
#         self.shift_range = shift_range
#     def __call__(self, point_data):
#         shifts = np.random.uniform(-self.shift_range, self.shift_range, (1,3))
#         point_data += shifts
#         return point_data
#
#
# class random_scale_point_cloud(Augmenter):
#     """ Randomly scale the point cloud. Scale is per point cloud.
#         Input:
#             Nx3 array, original point clouds
#         Return:
#             Nx3 array, scaled point clouds
#     """
#     def __init__(self,  scale_low=0.8, scale_high=1.25):
#         super(random_scale_point_cloud, self).__init__()
#         self.scale_low = scale_low
#         self.scale_high = scale_high
#     def __call__(self, point_data):
#         scales = np.random.uniform(self.scale_low, self.scale_high, 1)
#         point_data *= scales
#         return point_data
#
# class random_point_dropout(Augmenter):
#     ''' batch_pc: Nx3 '''
#     def __init__(self, max_dropout_ratio=0.875):
#         super(random_point_dropout, self).__init__()
#         self.max_dropout_ratio=max_dropout_ratio
#     def __call__(self, point_data):
#         dropout_ratio =  np.random.random()*self.max_dropout_ratio # 0~0.875
#         drop_idx = np.where(np.random.random((point_data.shape[0]))<=dropout_ratio)[0]
#         if len(drop_idx)>0:
#             point_data[drop_idx,:] = point_data[0,:] # set to the first point
#         return point_data
#
# class normalize_point_cloud(Augmenter):
#     def __init__(self, mean=None, std=None):
#         super(normalize_point_cloud, self).__init__()
#         self.mean = mean
#         self.std = std
#     def __call(self, point_data):
#         if self.mean is None:
#             self.mean = np.mean(point_data, axis=0)
#         if self.std is None:
#             self.std = np.max(np.sqrt(np.sum((point_data - self.mean)**2, axis=1)))
#         return (point_data-self.mean)/self.std

def normalize_point_cloud(batch_data, mean=None, std=None):
        if mean is None:
            mean = np.mean(batch_data, axis=0)
        if std is None:
            std = np.max(np.sqrt(np.sum((batch_data - mean)**2, axis=1)))
        return (batch_data-mean)/std

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(batch_data, rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_data = np.dot(batch_data, rotation_matrix)
    return rotated_data

# def rotate_point_cloud_with_normal(batch_xyz_normal):
#     ''' Randomly rotate XYZ, normal point cloud.
#         Input:
#             batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
#         Output:
#             B,N,6, rotated XYZ, normal point cloud
#     '''
#     for k in xrange(batch_xyz_normal.shape[0]):
#         rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(rotation_angle)
#         sinval = np.sin(rotation_angle)
#         rotation_matrix = np.array([[cosval, 0, sinval],
#                                     [0, 1, 0],
#                                     [-sinval, 0, cosval]])
#         shape_pc = batch_xyz_normal[k,:,0:3]
#         shape_normal = batch_xyz_normal[k,:,3:6]
#         batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
#         batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
#     return batch_xyz_normal

# def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
#     """ Randomly perturb the point clouds by small rotations
#         Input:
#           BxNx6 array, original batch of point clouds and point normals
#         Return:
#           BxNx3 array, rotated batch of point clouds
#     """
#     rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
#     for k in xrange(batch_data.shape[0]):
#         angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
#         Rx = np.array([[1,0,0],
#                        [0,np.cos(angles[0]),-np.sin(angles[0])],
#                        [0,np.sin(angles[0]),np.cos(angles[0])]])
#         Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
#                        [0,1,0],
#                        [-np.sin(angles[1]),0,np.cos(angles[1])]])
#         Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
#                        [np.sin(angles[2]),np.cos(angles[2]),0],
#                        [0,0,1]])
#         R = np.dot(Rz, np.dot(Ry,Rx))
#         shape_pc = batch_data[k,:,0:3]
#         shape_normal = batch_data[k,:,3:6]
#         rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
#         rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
#     return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(batch_data, rotation_matrix)
    return rotated_data

# def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
#     """ Rotate the point cloud along up direction with certain angle.
#         Input:
#           BxNx6 array, original batch of point clouds with normal
#           scalar, angle of rotation
#         Return:
#           BxNx6 array, rotated batch of point clouds iwth normal
#     """
#     rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
#     for k in xrange(batch_data.shape[0]):
#         #rotation_angle = np.random.uniform() * 2 * np.pi
#         cosval = np.cos(rotation_angle)
#         sinval = np.sin(rotation_angle)
#         rotation_matrix = np.array([[cosval, 0, sinval],
#                                     [0, 1, 0],
#                                     [-sinval, 0, cosval]])
#         shape_pc = batch_data[k,:,0:3]
#         shape_normal = batch_data[k,:,3:6]
#         rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
#         rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
#     return rotated_data



def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)

    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    rotated_data = np.dot(batch_data, R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    shifts = np.random.uniform(-shift_range, shift_range, (1,3))
    batch_data += shifts
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high, 1)
    batch_data *= scales
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''

    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        batch_pc[drop_idx,:] = batch_pc[0,:] # set to the first point
    return batch_pc


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)