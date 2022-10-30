import cv2
import numpy as np

ALPHA = 1
BHETA = 1 * np.pi / 180
RESOLUTION = 0.1
MAX_RANGE = 50
MAP_WIDTH = 100
SPHERICAL2CARTESIAN_BIAS = 6
MAP_SIZE_X = int(MAP_WIDTH / RESOLUTION)
MAP_SIZE_Y = int(MAP_WIDTH / RESOLUTION)

_XARR = np.arange(-MAP_WIDTH / 2, MAP_WIDTH / 2, RESOLUTION)
_YARR = np.arange(-MAP_WIDTH / 2, MAP_WIDTH / 2, RESOLUTION)
_MAP_XX, _MAP_YY = np.meshgrid(_XARR, -_YARR)
_RGRID = np.sqrt(np.add(np.square(_MAP_XX), np.square(_MAP_YY)))
OOR_MASK = _RGRID >= MAX_RANGE


def init_ogm():
    return np.ones((MAP_SIZE_Y, MAP_SIZE_X)) * 0.5


def generate_measurement_ogm(lidar_in, ogm_shape):
    rphi_meas = np.zeros((lidar_in.shape[0], 2))
    rphi_meas[:, 1] = (
        np.sqrt(np.add(np.square(lidar_in[:, 0]), np.square(lidar_in[:, 1]))) / ALPHA
    )
    rphi_meas[:, 0] = (np.arctan2(lidar_in[:, 1], lidar_in[:, 0]) + np.pi) / BHETA
    rphi_meas = np.unique(rphi_meas.astype(int), axis=0)
    rphi_meas = rphi_meas[rphi_meas[:, 1] < int(MAX_RANGE / ALPHA), :]
    rphi_meas = rphi_meas[rphi_meas[:, 0] < int(2 * np.pi / BHETA), :]

    sg_ang_bin = int(2 * np.pi / BHETA)
    sg_rng_bin = int(MAX_RANGE / ALPHA)
    scan_grid = np.ones((sg_ang_bin, sg_rng_bin)) * 0.5
    scan_grid[tuple(rphi_meas.T)] = 0.7

    for ang in range(sg_ang_bin):
        ang_arr = rphi_meas[rphi_meas[:, 0] == ang, 1]
        if len(ang_arr) == 0:
            scan_grid[ang, :] = 0.3
        else:
            min_r = np.min(ang_arr)
            scan_grid[ang, :min_r] = 0.3

    ogm_sz = (ogm_shape[1], ogm_shape[0])
    ogm_cen = (int(ogm_shape[1] / 2), int(ogm_shape[0] / 2))
    radius = (MAX_RANGE / RESOLUTION) + SPHERICAL2CARTESIAN_BIAS
    ogm_step = cv2.warpPolar(scan_grid, ogm_sz, ogm_cen, radius, cv2.WARP_INVERSE_MAP)
    ogm_step[OOR_MASK] = 0.5
    ogm_step = cv2.rotate(ogm_step, cv2.ROTATE_90_CLOCKWISE)
    return ogm_step


def logit(m):
    return np.log(np.divide(m, np.subtract(1, m)))


def inverse_logit(m):
    return np.divide(np.exp(m), np.add(1, np.exp(m)))


def update_ogm(prior_ogm, new_ogm):
    logit_map = logit(new_ogm) + logit(prior_ogm)
    out_ogm = inverse_logit(logit_map)
    out_ogm[out_ogm >= 0.98] = 0.98
    out_ogm[out_ogm <= 0.02] = 0.02
    return out_ogm


def shift_pose_ogm(ogm, init, fin):
    ogm_o = ogm.copy()
    theta = init[2] / 180 * np.pi
    rot_m = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    trs_m = np.array([[init[0]], [init[1]]])
    point = np.array(fin[:2]).reshape((-1, 1))
    point_1 = np.subtract(point, trs_m)
    point_2 = np.dot(rot_m, -point_1)
    delta_theta = fin[2] - init[2]
    delta = np.array([point_2[1, 0] / RESOLUTION, point_2[0, 0] / RESOLUTION, 0])

    M = np.array([[1, 0, delta[0]], [0, 1, -delta[1]]])
    dst = cv2.warpAffine(ogm_o, M, (ogm_o.shape[1], ogm_o.shape[0]), borderValue=0.5)
    M = cv2.getRotationMatrix2D(
        (ogm_o.shape[1] / 2 + 0.5, ogm_o.shape[0] / 2 + 0.5), delta_theta, 1
    )
    dst = cv2.warpAffine(dst, M, (ogm_o.shape[1], ogm_o.shape[0]), borderValue=0.5)
    return dst
