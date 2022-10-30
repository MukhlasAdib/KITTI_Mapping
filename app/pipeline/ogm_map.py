from pystream import Stage

from libs.sensors import filter_road_points, get_road_model_ransac
from libs.ogm import (
    init_ogm,
    shift_pose_ogm,
    generate_measurement_ogm,
    update_ogm,
)
from pipeline.data import AppData


class OGMMapStage(Stage):
    def __init__(self, road_height_threshold):
        self.mapper = OGMMap(road_height_threshold)

    def __call__(self, data: AppData):
        data.output_data.ogm = self.mapper.update(
            data.output_data.segmentation_pred,
            data.input_data.lidar_2d,
            data.input_data.lidar_in_cam,
            data.input_data.lidar_raw,
            data.input_data.pose,
            data.input_data.previous_pose,
        )
        return data

    def cleanup(self):
        pass


class OGMMap:
    def __init__(self, road_height_threshold):
        self.road_height_threshold = road_height_threshold
        self.ogm = init_ogm()

    def update(self, segm_pred, lidar_2d, lidar_in_cam, lidar_raw, pose, old_pose):
        road_model = get_road_model_ransac(segm_pred, lidar_in_cam, lidar_2d)
        lidar_nonroad = filter_road_points(
            road_model, lidar_raw, self.road_height_threshold
        )
        lidar_ogm = lidar_nonroad[:, [2, 0]]

        shifted_ogm = shift_pose_ogm(self.ogm, old_pose, pose)
        ogm_step = generate_measurement_ogm(lidar_ogm, self.ogm.shape)
        self.ogm = update_ogm(shifted_ogm, ogm_step)
        return self.ogm
