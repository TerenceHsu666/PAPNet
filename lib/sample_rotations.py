import logging
import math
import numpy as np
from lib.transformations import quaternion_from_matrix


def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def sample_rotations_12():
    """ tetrahedral_group: 12 rotations

    """
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]])
    # return group.astype(float)
    quaternion_group = np.zeros((12, 4))
    for i in range(12):
        quaternion_group[i] = quaternion_from_matrix(group[i])
    return quaternion_group.astype(float)


def sample_rotations_24():
    """ octahedral_group: 24 rotations

    """
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                      [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
                      [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
                      [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],

                      [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
                      [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
                      [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
                      [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],

                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                      [[0, 0, 1], [0, -1, 0], [-1, 0, 0]],
                      [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
                      [[0, 0, -1], [0, -1, 0], [1, 0, 0]]])
    # return group.astype(float)
    quaternion_group = np.zeros((24, 4))
    for i in range(24):
        quaternion_group[i] = quaternion_from_matrix(group[i])
    return quaternion_group.astype(float)


def sample_rotations_60(return_type="quaternion"):
    """ icosahedral_group: 60 rotations
        args:
            return_type: str "matrix" or int 0
                         str "quaternion" or int 1
    """
    phi = (1 + math.sqrt(5)) / 2
    R1 = np.array([[-phi/2, 1/(2*phi), -0.5], [-1/(2*phi), 0.5, phi/2], [0.5, phi/2, -1/(2*phi)]])
    R2 = np.array([[phi/2, 1/(2*phi), -0.5], [1/(2*phi), 0.5, phi/2], [0.5, -phi/2, 1/(2*phi)]])
    group = [np.eye(3, dtype=float)]
    n = 0
    while len(group) > n:
        n = len(group)
        set_so_far = group
        for rot in set_so_far:
            for R in [R1, R2]:
                new_R = np.matmul(rot, R)
                new = True
                for item in set_so_far:
                    if np.sum(np.absolute(item - new_R)) < 1e-6:
                        new = False
                        break
                if new:
                    group.append(new_R)
                    break
            if new:
                break

    if return_type == "matrix" or return_type == 0:
        return np.array(group)

    elif return_type == "quaternion" or return_type == 1:
        group = np.array(group)
        quaternion_group = np.zeros((60, 4))
        for i in range(60):
            quaternion_group[i] = quaternion_from_matrix(group[i])
        return quaternion_group.astype(float)

    else:
        raise ValueError('Unknown return rotation type')


def grid_xyz(xy_bin_num=20, z_bin_num=40, xy_bin_range=(-200, 200), z_bin_range=(0.0, 2.0),):
    bin_size_xy = (xy_bin_range[1]-xy_bin_range[0])/ (2 * xy_bin_num)
    xy_bin_ctrs = np.linspace(xy_bin_range[0], xy_bin_range[1],xy_bin_num, endpoint=False) + bin_size_xy
    
    bin_size_z = (z_bin_range[1]-z_bin_range[0])/ (2 * z_bin_num)
    z_bin_ctrs = np.linspace(z_bin_range[0], z_bin_range[1],z_bin_num, endpoint=False) + bin_size_z

    return xy_bin_ctrs, z_bin_ctrs
