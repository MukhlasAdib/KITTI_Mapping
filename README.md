# MAPPING ON KITTI DATASET - A TUTORIAL

## Overview

This repo contains tutorials on autonomous vehicle's mapping algorithm on KITTI Dataset. This tutorial demonstrate two examples of grid-based mapping technique, which are occupancy grid map (OGM) and evidential dynamic grid map (DGM). The sensors used are monocular camera and LiDAR.

- For detailed step-by-step (including theory explanation) are provided in the [step-by-step mapping tutorial notebook](https://github.com/MukhlasAdib/KITTI_Mapping/blob/main/KITTI_Mapping_Tutorial_Step_by_step.ipynb).
- To see how the algorithms work on the full data sequence, you can run the [full loop mapping tutorial notebook](https://github.com/MukhlasAdib/KITTI_Mapping/blob/main/KITTI_Mapping_Tutorial_Full_Loop.ipynb).
- I also have the `.py` script to run the algorithms. Please go to `app` and run "`python run.py`". Before that, please install the requirements by using "`pip install -r requirements.txt`" from `app`. The script uses PyStream as the pipeline constructor to boost the pipeline throughput. Please check the `pystream-pipeline` project [here](https://github.com/MukhlasAdib/pystream-pipeline).

In case you are not familiar with grid map, here is an example of it. As can be seen, we aim to map regions around the vehicle and find which areas are drivable (white pixels) and which are not (black/gray pixels)

<center><img src="https://github.com/MukhlasAdib/KITTI_Mapping/blob/main/figures/2_0_3443.png?raw=true" width=400px></center>

## Results

Here are results from the full loop notebook

### OGM

<center><img src="https://github.com/MukhlasAdib/KITTI_Mapping/blob/main/figures/results_ogm_min.gif?raw=true" width=800px></center>

### DGM

\*The dynamic cells are colored blue

<center><img src="https://github.com/MukhlasAdib/KITTI_Mapping/blob/main/figures/results_dgm.gif?raw=true" width=800px></center>

## References

[1] Repository for this tutorial: [here](https://github.com/MukhlasAdib/KITTI_Mapping).

[2] The full KITTI datased can be accessed [here](http://www.cvlibs.net/datasets/kitti/).

[3] KITTI Dataset paper: A. Geiger, P. Lenz, C. Stiller and R. Urtasun, "Vision meets Robotics: The KITTI Dataset," *International Journal of Robotics Research (IJRR)*, vol. 32, no. 11, pp. 1231-1237 2013.

[4] Description of Occupancy Grid Map (OGM) estimation: Z. Luo, M. V. Mohrenschilt and S. Habibi, "A probability occupancy grid based approach for real-time LiDAR ground segmentation," *IEEE Transactions on Intelligent Transportation Systems*, vol 21, no. 3, pp. 998–1010, Mar. 2020.

[5] Description of Dynamic Grid Map (DGM) estimation: J. Moras, V. Cherfaoui and P. Bonnifait, "Credibilist occupancy grids for vehicle perception in dynamic environments," *2011 IEEE International Conference on Robotics and Automation*, Shanghai, China, 2011, pp. 84-89.

[6] Paper of DeepLab v3+ for image segmentation: L. C. Chen, Y. Zhu, G. Apandreou, F. Schroff and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” *ECCV 2018 Lecture Notes in Computer Science*, vol. 11211, pp. 833–851, 2018.

[7] DeepLab v3+ paper via arXiv: [here](https://arxiv.org/abs/1802.02611).

[8] DeepLab v3+ repository: [here](https://github.com/tensorflow/models/tree/master/research/deeplab).

[9] This tutorial use pykitti module to load the KITTI dataset: [here](https://github.com/utiasSTARS/pykitti).

[10] The script version uses pystream-pipeline as the real-time pipeline constructor: [here](https://github.com/MukhlasAdib/pystream-pipeline)
