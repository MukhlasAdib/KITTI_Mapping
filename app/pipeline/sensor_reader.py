from pystream import Stage

from pipeline.data import AppData
from libs.sensors import (
    crop_data,
    load_data,
    load_vehicle_pose_vel,
    project_lidar2cam,
    transform_coordinate,
)


class SensorReaderStage(Stage):
    def __init__(self, kitti_data, crop_ratio_w, crop_ratio_h):
        self.reader = SensorReader(kitti_data, crop_ratio_w, crop_ratio_h)

    def __call__(self, data: AppData):
        data.input_data.previous_pose = self.reader.pose
        (
            data.input_data.input_image,
            data.input_data.lidar_in_cam,
            data.input_data.lidar_2d,
            data.input_data.lidar_raw,
            data.input_data.pose,
        ) = self.reader.get_data(
            data.metadata.data_idx,
            data.metadata.previous_data_idx,
        )
        return data

    def cleanup(self):
        pass


class SensorReader:
    def __init__(self, kitti_data, crop_ratio_w, crop_ratio_h):
        self.crop_ratio_w = crop_ratio_w
        self.crop_ratio_h = crop_ratio_h
        self.kitti_data = kitti_data
        self.lidar2cam_extrinsic = self.kitti_data.calib.T_cam2_velo
        self.camera_intrinsic = self.kitti_data.calib.K_cam2
        self.pose = (0, 0, 0)

    def get_data(self, data_idx, previous_idx):
        img_raw, lidar_raw = load_data(self.kitti_data, data_idx)
        img_raw_size = img_raw.shape
        lidar_raw = transform_coordinate(lidar_raw, self.lidar2cam_extrinsic)
        lidar_2d, lidar_in_cam = project_lidar2cam(
            lidar_raw, self.camera_intrinsic, img_raw_size
        )
        crop_img, lidar_2d, lidar_in_cam = crop_data(
            img_raw, lidar_2d, lidar_in_cam, self.crop_ratio_h, self.crop_ratio_w
        )
        self.pose = load_vehicle_pose_vel(
            self.kitti_data, data_idx, self.pose, previous_idx
        )
        return crop_img, lidar_in_cam, lidar_2d, lidar_raw, self.pose
