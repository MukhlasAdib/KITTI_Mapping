import os
import time

import pykitti
from pystream import Pipeline
import tqdm

from pipeline.data import AppData
from pipeline.road_segmentation import RoadSegmentationStage
from pipeline.sensor_reader import SensorReaderStage
from pipeline.ogm_map import OGMMapStage
from pipeline.output_writer import OutputWriterStage

KITTI_DIR = "../raw_data/"
KITTI_DATE = "2011_09_26"
KITTI_DRIVE = "0013"
CROP_RH = 3
CROP_RW = 4

MODEL_PATH = "../pretrained/deeplab_model.pb"
MODEL_INPUT_SIZE = 513

ROAD_HEIGHT_THRESHOLD = 0.15

OUTPUT_DIR = "results"


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    kitti_data = pykitti.raw(KITTI_DIR, KITTI_DATE, KITTI_DRIVE)
    sensor_reader_stage = SensorReaderStage(kitti_data, CROP_RH, CROP_RH)
    road_segmentation_stage = RoadSegmentationStage(MODEL_PATH, MODEL_INPUT_SIZE)
    ogm_map_stage = OGMMapStage(ROAD_HEIGHT_THRESHOLD)
    output_writer_stage = OutputWriterStage(OUTPUT_DIR)

    app_pipeline = Pipeline()
    app_pipeline.add(sensor_reader_stage)
    app_pipeline.add(road_segmentation_stage)
    app_pipeline.add(ogm_map_stage)
    app_pipeline.add(output_writer_stage)
    app_pipeline.parallelize()
    # app_pipeline.serialize()

    number_data = len(kitti_data.oxts)
    old_idx = 0
    for idx in tqdm.tqdm(range(number_data)):
        app_data = AppData()
        app_data.metadata.data_idx = idx
        app_data.metadata.previous_data_idx = old_idx
        app_pipeline.forward(app_data)
        old_idx = idx
    time.sleep(10)
    app_pipeline.cleanup()


if __name__ == "__main__":
    main()
