import numpy as np
from pystream import Stage
import tensorflow as tf

from libs.sensors import segmentation_inference
from pipeline.data import AppData


class RoadSegmentationStage(Stage):
    def __init__(self, model_path, input_size):
        self.model = RoadSegmentation(model_path, input_size)

    def __call__(self, data: AppData):
        data.output_data.segmentation_pred = self.model.infer(
            data.input_data.input_image
        )
        return data

    def cleanup(self):
        pass


class RoadSegmentation:
    def __init__(self, model_path, input_size):
        self.input_size = input_size
        with open(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef.FromString(f.read())  # type: ignore
        graph = tf.Graph()
        with graph.as_default():  # type: ignore
            tf.import_graph_def(graph_def=graph_def, name="")
        self.infer_sess = tf.compat.v1.Session(graph=graph)
        segmentation_inference(
            np.ones((600, 600, 3), dtype=np.uint8),
            self.infer_sess,
            self.input_size,
            0.5,
        )

    def infer(self, input_image):
        _, segm_pred = segmentation_inference(
            input_image, self.infer_sess, self.input_size, 0.5
        )
        return segm_pred
