from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Metadata:
    data_idx: int = 0
    previous_data_idx: int = 0


@dataclass
class InputData:
    input_image: np.ndarray = np.array([])
    lidar_in_cam: np.ndarray = np.array([])
    lidar_2d: np.ndarray = np.array([])
    lidar_raw: np.ndarray = np.array([])
    pose: Tuple = (0, 0, 0)
    previous_pose: Tuple = (0, 0, 0)


@dataclass
class OutputData:
    segmentation_pred: np.ndarray = np.array([])
    ogm: np.ndarray = np.array([])


@dataclass
class AppData:
    metadata: Metadata = Metadata()
    input_data: InputData = InputData()
    output_data: OutputData = OutputData()
