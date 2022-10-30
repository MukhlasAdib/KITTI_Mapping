# Mapping on KITTI Dataset - Script Version

This is the script implementation of the KITTI mapping pipeline. This implementation is made as an example of PyStream implementation. PyStream is a Python pipeline manager that can help you construct a data pipeline for real-time application. The package is able to parallelize your pipeline and boost the the throughput of it. For now, the implementation is only available for OGM. When using serial pipeline mode, I noted that the throughput of the output is around 2.5 FPS in my laptop\*. When using the parallel mode of PyStream, the throughput is improved to 3.5 FPS.

*\*11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, 8 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU.*

The source code is structured as follows:

- `libs` folder contains base modules that are copied from the notebook version of the pipeline.
- `pipeline` is the implementation of the PyStream stages. Please check this folder to see how PyStream stages can be made.
- `run.py` is the main application script. Please also check the file to see how to construct a PyStream pipeline.

To run the application, please run the following command from this folder

```bash
python run.py
```
