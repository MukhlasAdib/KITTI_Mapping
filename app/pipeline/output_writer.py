import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pystream import Stage

from pipeline.data import AppData

matplotlib.use("Agg")


class OutputWriterStage(Stage):
    def __init__(self, save_dir):
        self.writer = OutputWriter(save_dir)
        self.qps = []
        self.prev_end_time = time.perf_counter()

    def __call__(self, data: AppData):
        self.writer.visualize(
            data.metadata.data_idx, data.output_data.ogm, data.input_data.input_image
        )
        self.update_qps()
        return data

    def update_qps(self):
        end_time = time.perf_counter()
        delta = end_time - self.prev_end_time
        self.qps.append(delta)
        self.prev_end_time = end_time

    def cleanup(self):
        avg = np.mean(self.qps[1:])
        print(f"Average throughput {1/ avg}")


class OutputWriter:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def visualize(self, idx, ogm, camera_img):
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
        plt.savefig(f"{self.save_dir}/{idx:03d}.png")
        plt.close(fig)
