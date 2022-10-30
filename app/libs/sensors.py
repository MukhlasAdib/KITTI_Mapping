import cv2
import numpy as np
from scipy import stats
from sklearn.linear_model import RANSACRegressor


def load_data(data, idx):
    img_raw = np.array(data.get_cam2(idx))
    lidar_raw = np.array(data.get_velo(idx))[:, :3]
    lidar_raw = lidar_raw[lidar_raw[:, 2] <= 0, :]
    dist = np.linalg.norm(lidar_raw, axis=1)
    lidar_raw = lidar_raw[dist >= 2.5]
    return img_raw, lidar_raw


def transform_coordinate(lidar_points, extrinsic_matrix):
    inp = lidar_points.copy()
    inp = np.concatenate((inp, np.ones((inp.shape[0], 1))), axis=1)
    inp = np.matmul(extrinsic_matrix, inp.T).T
    return inp[:, :3]


def project_lidar2cam(lidar_in_cam, camera_intrinsic, img_raw_size):
    lidar_in_cam = np.concatenate(
        (lidar_in_cam, np.ones((lidar_in_cam.shape[0], 1))), axis=1
    )
    lidar_in_cam = lidar_in_cam[lidar_in_cam[:, 2] > 0]

    lidar_2d = np.matmul(camera_intrinsic, lidar_in_cam[:, :3].T).T
    lidar_2d = np.divide(lidar_2d, lidar_2d[:, 2].reshape((-1, 1)))
    lidar_2d = lidar_2d.astype(int)

    maskH = np.logical_and(lidar_2d[:, 0] >= 0, lidar_2d[:, 0] < img_raw_size[1])
    maskV = np.logical_and(lidar_2d[:, 1] >= 0, lidar_2d[:, 1] < img_raw_size[0])
    mask = np.logical_and(maskH, maskV)
    lidar_2d = lidar_2d[mask, :]
    lidar_in_cam = lidar_in_cam[mask, :]

    return lidar_2d, lidar_in_cam[:, :3]


def crop_data(img_in, lidar_2d_in, lidar_in_cam_in, rh, rw):
    lidar_2d = lidar_2d_in.copy()
    lidar_in_cam = lidar_in_cam_in.copy()
    img = img_in.copy()

    dim_ori = np.array(img.shape)
    cent = (dim_ori / 2).astype(int)
    if dim_ori[0] / dim_ori[1] == rh / rw:
        crop_img = img
        cW = int(dim_ori[1] / 2)
        cH = int(dim_ori[0] / 2)

    elif dim_ori[0] <= dim_ori[1]:
        cH2 = dim_ori[0]
        cW2 = cH2 * rw / rh
        cH = int(cH2 / 2)
        cW = int(cW2 / 2)
        crop_img = img[:, cent[1] - cW : cent[1] + cW + 1]

    else:
        cW2 = dim_ori[1]
        cH2 = cW2 * rh / rw
        cW = int(cW2 / 2)
        cH = int(cH2 / 2)
        crop_img = img[cent[0] - cH : cent[0] + cH + 1, :]

    centH = cent[0]
    centW = cent[1]
    maskH = np.logical_and(lidar_2d[:, 1] >= centH - cH, lidar_2d[:, 1] <= centH + cH)
    maskW = np.logical_and(lidar_2d[:, 0] >= centW - cW, lidar_2d[:, 0] <= centW + cW)
    mask = np.logical_and(maskH, maskW)
    lidar_2d = lidar_2d[mask, :]
    lidar_in_cam = lidar_in_cam[mask, :]
    cent = np.array((centW - cW, centH - cH, 0)).reshape((1, 3))
    lidar_2d = lidar_2d - cent

    return crop_img, lidar_2d.astype(int), lidar_in_cam


def segmentation_inference(img_in, sess, target_size=513, probability_threshold=0.5):
    INPUT_TENSOR_NAME = "ImageTensor:0"
    PROB_TENSOR_NAME = "SemanticProbabilities:0"
    INPUT_SIZE = target_size

    image = img_in.copy()
    sz = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if INPUT_SIZE == 0:
        resized_image = image.copy()
    else:
        resized_image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))

    batch_seg_map = sess.run(
        PROB_TENSOR_NAME, feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
    )
    seg_map = (batch_seg_map[0][:, :, 1] * 255).astype(int)
    prob = np.array(seg_map, dtype=np.uint8)
    prob = cv2.resize(prob, (sz[1], sz[0]))
    pred = prob.copy()

    msk_bin = np.greater_equal(prob, (probability_threshold * 255))
    pred[msk_bin] = 1
    pred[np.logical_not(msk_bin)] = 0

    _, segm_reg = cv2.connectedComponents(pred)
    segm_reg = segm_reg.astype(float)
    segm_reg[segm_reg == 0] = np.nan
    modes, _ = stats.mode(segm_reg.flatten(), keepdims=True, nan_policy="omit")
    mode = modes[0]
    pred[segm_reg != mode] = 0

    return prob, np.multiply(pred, 255).astype(np.uint8)


def get_road_model_ransac(img_pred, lidar_in_cam, lidar_2d):
    lidar_in_road_lbl = [
        True if img_pred[pt[1], pt[0]] == 255 else False for pt in lidar_2d
    ]
    lidar_in_road = lidar_in_cam[lidar_in_road_lbl, :]
    road_model = RANSACRegressor().fit(lidar_in_road[:, [0, 2]], lidar_in_road[:, 1])
    return road_model


def filter_road_points(road_model, lidar_in, threshold=0.5):
    x = lidar_in[:, [0, 2]]
    y_true = lidar_in[:, 1]
    y_pred = road_model.predict(x)
    delta_y = np.absolute(y_true - y_pred).flatten()
    is_not_road = delta_y > threshold
    lidar_out = lidar_in[is_not_road, :].copy()
    return lidar_out


def load_vehicle_pose_vel(data, idx, old_pose, old_idx):
    delta_t = (data.timestamps[idx] - data.timestamps[old_idx]).total_seconds()
    packet = data.oxts[idx].packet
    vf = packet.vf
    vr = -packet.vl
    pose_f = old_pose[0] + (vf * delta_t)
    pose_r = old_pose[1] + (vr * delta_t)
    pose_y = packet.yaw - data.oxts[0].packet.yaw
    return (pose_f, pose_r, pose_y)
