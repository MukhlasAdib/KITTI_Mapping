import os
import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pykitti
import tensorflow as tf

from libs.sensors import (
    crop_data,
    filter_road_points,
    get_road_model_ransac,
    load_data,
    load_vehicle_pose_vel,
    process_images,
    project_lidar2cam,
    transform_coordinate,
)
from libs.ogm import (
    MAP_SIZE_X,
    MAP_SIZE_Y,
    shift_pose_ogm,
    generate_measurement_ogm,
    update_ogm,
)

KITTI_DIR = "../raw_data/"
KITTI_DATE = "2011_09_26"
KITTI_DRIVE = "0013"

OUTPUT_DIR = "results"

CROP_RH = 3
CROP_RW = 4
DEEPLAB_MODEL_PATH = "../pretrained/deeplab_model.pb"
DEEPLAB_INPUT_SIZE = 513
ROAD_HEIGHT_THRESHOLD = 0.15


class Mapper:
    def __init__(self, data):
        self.data = data
        self.number_data = len(data.oxts)
        self.lidar2cam_extrinsic = data.calib.T_cam2_velo
        self.camera_intrinsic = data.calib.K_cam2

        self.old_idx = 0
        self.old_pose = (0, 0, 0)
        self.load_model()

    def load_model(self):
        with open(DEEPLAB_MODEL_PATH, "rb") as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())  # type: ignore
        graph = tf.Graph()
        with graph.as_default():  # type: ignore
            tf.import_graph_def(graph_def=graph_def, name="")
        self.infer_sess = tf.compat.v1.Session(graph=graph)
        process_images(
            np.ones((600, 600, 3), dtype=np.uint8),
            self.infer_sess,
            DEEPLAB_INPUT_SIZE,
            0.5,
        )

    def single_loop_ogm(self, idx, tf_sess, ogm):
        img_raw, lidar_raw = load_data(self.data, idx)
        img_raw_size = img_raw.shape
        lidar_raw = transform_coordinate(lidar_raw, self.lidar2cam_extrinsic)
        lidar_2d, lidar_in_cam = project_lidar2cam(
            lidar_raw, self.camera_intrinsic, img_raw_size
        )
        crop_img, lidar_2d, lidar_in_cam = crop_data(
            img_raw, lidar_2d, lidar_in_cam, CROP_RH, CROP_RW
        )
        _, segm_pred = process_images(crop_img, tf_sess, DEEPLAB_INPUT_SIZE, 0.5)
        road_model = get_road_model_ransac(segm_pred, lidar_in_cam, lidar_2d)
        lidar_nonroad = filter_road_points(road_model, lidar_raw, ROAD_HEIGHT_THRESHOLD)
        lidar_ogm = lidar_nonroad[:, [2, 0]]

        pose = load_vehicle_pose_vel(self.data, idx, self.old_pose, self.old_idx)
        shifted_ogm = shift_pose_ogm(ogm, self.old_pose, pose)
        ogm_step = generate_measurement_ogm(lidar_ogm, ogm.shape)
        updated_ogm = update_ogm(shifted_ogm, ogm_step)

        return updated_ogm, pose, crop_img

    def run(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ### Initiate OGM
        ogm = np.ones((MAP_SIZE_Y, MAP_SIZE_X)) * 0.5

        ### Process all the data in sequence
        idx = 0
        self.old_idx = 0
        self.old_pose = (0, 0, 0)
        frequency = 1

        for idx in tqdm.tqdm(range(self.number_data)):
            ogm, pose, camera_img = self.single_loop_ogm(idx, self.infer_sess, ogm)
            self.old_idx = idx
            self.old_pose = pose
            idx = idx + frequency

            ### Visualize
            fig, axs = plt.subplots(1, 2, figsize=(16, 8))
            ogm_img = ((1 - ogm) * 255).astype(np.uint8)
            ogm_img = cv2.resize(ogm_img, (500, 500))
            ogm_img = cv2.cvtColor(ogm_img, cv2.COLOR_GRAY2RGB)
            center = (int(ogm_img.shape[1] / 2), int(ogm_img.shape[0] / 2))
            cv2.circle(ogm_img, tuple(center[:2]), 5, (255, 0, 0), -1)
            axs[0].imshow(ogm_img, cmap="gray", vmin=0, vmax=255)  # type: ignore
            axs[1].imshow(camera_img)  # type: ignore
            axs[0].set_axis_off()  # type: ignore
            axs[1].set_axis_off()  # type: ignore
            plt.savefig(f"{save_dir}/{self.old_idx:03d}.png")
            plt.close(fig)


def main():
    data = pykitti.raw(KITTI_DIR, KITTI_DATE, KITTI_DRIVE)
    mapper = Mapper(data)
    mapper.run(OUTPUT_DIR)


if __name__ == "__main__":
    main()
