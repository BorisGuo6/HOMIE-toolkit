# HOMIE-toolkit

Tools for **reading** and **visualizing** [Xperience-10M](https://huggingface.co/datasets/ropedia-ai/xperience-10m) data.

- Load annotation and use the data in your own scripts (export, training, custom viz)
- Reuse visualization helpers (depth colormap, skeleton, point cloud) with Rerun

## Layout

- **`data_loader.py`** – Load `annotation.hdf5`: calibration, SLAM poses, hand/body mocap, depth, IMU, point cloud. List HDF5 contents and load video frames.
- **`visualization.py`** – `create_blueprint`, `depth_to_colormap`, `depth_to_pointcloud`, `build_line3d_skeleton`, etc.
- **`examples/`**
  - **`example_load_annotation.py`** – How to read data: list contents, load annotation, inspect calibration.
  - **`example_visualize_rrd.py`** – Load data with `data_loader`, then log skeleton + depth to a Rerun `.rrd` file.

## Dependencies

```bash
conda create -n homie python=3.12
conda activate homie
pip install -r requirements.txt
```

## Run examples (from package root)

Download sample data [here](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample).

```bash
# List and load annotation
python examples/example_load_annotation.py --data_root /path/to/episode

# RRD: skeleton + depth for first N frames
python examples/example_visualize_rrd.py --data_root /path/to/episode --output_rrd vis.rrd
```

Then open the RRD:

```bash
rerun vis.rrd
```

## Data format (annotation.hdf5)

- **`calibration/`** – Same hierarchy as `calibration.json` (K, T_c_b, cam01, etc.).
- **`slam/`** – `quat_wxyz`, `trans_xyz`, `frame_names` (world-to-camera pose per frame).
- **`hand_mocap/`** – `left_joints_3d`, `right_joints_3d` (21 joints per hand, MANO order).
- **`full_body_mocap/`** – `keypoints` (SMPL-H 52 joints), `contacts`.
- **`depth/`** – `depth`, optional `confidence`, `scale`, `depth_min`, `depth_max`.
- **`imu/`** – `device_timestamp_ns`, `accel_xyz`, `gyro_xyz`, optional `keyframe_indices`.

See `example_load_annotation.py` and `list_annotation_contents()` for a quick way to inspect an episode.
