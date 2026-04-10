---
pretty_name: Xperience-10M
language:
- en
task_categories:
- video-classification
- image-to-text
- depth-estimation
- robotics
tags:
- egocentric
- first-person
- multimodal
- 3d
- 4d
- embodied-ai
- robotics
- human-motion
- mocap
- imu
- audio
- depth
- captions
- video
size_categories:
- 1M<n<10M

license: other

extra_gated_heading: "Request controlled access to Xperience-10M"
extra_gated_description: "Access is reviewed manually and is limited to approved non-commercial use. Completion of an external agreement-signing step may be required before approval. Please sign the agreement via [DocuSign](https://ropedia.docsend.com/view/ra7ej7gs6s98sw87)"
extra_gated_button_content: "I have signed the access control agreement"
---

# ⚠️ Important: If you have already submitted an access request but have not completed the required [DocuSign agreement](https://ropedia.docsend.com/view/ra7ej7gs6s98sw87), your request will remain pending. Please complete signing and we will grant access once verified.

<p align="center">
  <a href="https://ropedia.com/">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/69b116c65a198623dcbcc950/ilT9q1aoGwEL1xfZ09Ip2.png" alt="HOMIE-toolkit logo" width="400" />
  </a>
  <br />
  <em>Interactive Intelligence from Human Xperience</em>
</p>

# Xperience-10M

## Dataset Summary

**Xperience-10M** is a large-scale egocentric multimodal dataset of human experience for embodied AI, robotics, world models, and spatial intelligence.

It contains **10 million experiences (interaction)** and **10,000 hours** of synchronized first-person recordings with **six video streams, audio, stereo depth, camera pose, hand mocap, full-body mocap, IMU, and hierarchical language annotations**. With **2.88 billion RGB frames**, **720 million depth frames**, **576 million pose and mocap frames**, and **~1 PB** of total data, Xperience-10M is, to our knowledge, **by far the largest egocentric dataset with structured 3D/4D multimodal annotations**.

Xperience-10M is built for training and evaluating models that do not just see the world, but also understand motion, geometry, interaction, and embodied behavior as a unified stream of experience.

It is designed to support research and product development in:
- embodied AI
- world modeling
- robot learning from human experience
- egocentric perception
- action understanding
- multimodal foundation models
- human-object interaction
- sensor fusion
- 3D/4D scene and motion understanding
- real-to-sim and sim-to-real pipelines

Check out [xperience-10m-sample](https://huggingface.co/datasets/ropedia-ai/xperience-10m-sample) for data sample (coffee making!).

Check out [HOMIE-toolkit](https://github.com/Ropedia/HOMIE-toolkit) for sample code to load/visualize Xperience data.

![Xperience Annotations](https://cdn-uploads.huggingface.co/production/uploads/69b116c65a198623dcbcc950/a19FrgUd6vNJKNa07eHfc.png)

## What makes Xperience-10M different

Most existing egocentric datasets provide only a partial view of embodied experience: RGB video, sparse labels, or limited motion signals. Xperience-10M is designed differently.

It treats experience as a multimodal, structured, and temporally grounded signal. Each episode can include:
- what the wearer sees
- what the wearer hears
- how the camera moves through space
- how the hands move
- how the full body articulates
- what the depth geometry looks like
- what the IMU measures
- what task, subtask, action, interaction, and objects are involved

This makes Xperience-10M especially useful for building systems that learn from real human experience at scale.

## Supported Tasks and Use Cases

Xperience-10M can support a broad range of tasks, including but not limited to:

- egocentric action recognition
- task and subtask prediction
- action captioning
- temporal action localization
- human-object interaction understanding
- object grounding and recognition
- audio-visual learning
- visual-language pretraining
- embodied reasoning
- stereo and monocular depth estimation
- visual odometry and trajectory learning
- SLAM and camera pose estimation
- hand pose estimation
- body motion estimation
- multimodal sensor fusion
- imitation learning and behavior modeling
- policy learning for robotics
- world model training

## Languages

The language annotations are in **English**.

## Dataset Structure

Xperience-10M is organized as a collection of **episodes**. Each episode contains synchronized egocentric video files together with a unified `annotation.hdf5` file storing annotations, calibration, geometry, motion, inertial signals, and metadata.

### Episode Layout

A typical episode folder contains:

```text
episode/
├── fisheye_cam0.mp4  # fisheye camera 0
├── fisheye_cam1.mp4  # fisheye camera 1
├── fisheye_cam2.mp4  # fisheye camera 2
├── fisheye_cam3.mp4  # fisheye camera 3
├── stereo_left.mp4   # rectified stereo left
├── stereo_right.mp4  # rectified stereo right
└── annotation.hdf5   # all annotations and metadata
````

### Modalities

Each episode may include the following modalities:

* **Four fisheye video streams**
* **Two rectified stereo video streams**
* **Audio** aligned with all video streams
* **Stereo depth**
* **Camera pose / SLAM trajectory**
* **Two-hand motion capture**
* **Full-body motion capture**
* **IMU**
* **Episode metadata**
* **Hierarchical language captions**, including:

  * task
  * subtask
  * action
  * interaction
  * objects

### HDF5 Annotation Structure

The `annotation.hdf5` file stores synchronized annotations and metadata in the following structure:

```text
annotation.hdf5
├── calibration/
│   ├── cam0/
│   ├── cam1/
│   ├── cam2/
│   ├── cam3/
│   └── cam01/
├── slam/
│   ├── quat_wxyz
│   ├── trans_xyz
│   ├── frame_names
│   └── point_cloud
├── depth/
│   ├── depth
│   ├── confidence
│   ├── scale
│   ├── depth_min
│   └── depth_max
├── hand_mocap/
│   ├── left_joints_3d
│   ├── right_joints_3d
│   ├── left_translation
│   ├── right_translation
│   ├── left_mano_hand_pose
│   ├── right_mano_hand_pose
│   ├── left_mano_hand_global_orient
│   ├── right_mano_hand_global_orient
│   ├── left_mano_hand_betas
│   └── right_mano_hand_betas
├── full_body_mocap/
│   ├── keypoints
│   ├── contacts
│   ├── Ts_world_cpf
│   ├── Ts_world_root
│   ├── body_quats
│   ├── left_hand_quats
│   ├── right_hand_quats
│   ├── betas
│   └── frame_nums
├── imu/
│   ├── device_timestamp_ns
│   ├── accel_xyz
│   ├── gyro_xyz
│   └── keyframe_indices
├── video/
│   ├── device_timestamp
│   ├── frame_number
│   └── length_sec
├── metadata/
└── caption
```

## Annotation Details

Go checkout [HOMIE-toolkit](https://github.com/Ropedia/HOMIE-toolkit) for more tech details of our annotations.

## Key Statistics

| Statistic                             |         Value |
| ------------------------------------- | ------------: |
| Total number of experiences (interactions) |      **10M** |
| Video with audio                      |  **10,000 h** |
| RGB frames                            |     **2.88B** |
| Depth frames                          |      **720M** |
| Camera poses                          |      **576M** |
| Mocap frames                          |      **576M** |
| IMU frames                            |      **7.2B** |
| Caption sentences                     |       **16M** |
| Caption words                         |      **200M** |
| Caption vocabulary size               |        **6K** |
| Number of objects                     |      **350K** |
| Total storage                         |     **~1 PB** |
| Total trajectory length               | **39,000 km** |

## Uses

### Direct Use

Xperience-10M is intended for direct use in:

* multimodal pretraining
* egocentric perception research
* action understanding
* motion understanding
* 3D/4D reconstruction and tracking
* action-language grounding
* embodied foundation model training
* robotics and imitation learning
* world model training

### Out-of-Scope Use

Xperience-10M is not intended for:

* identity recognition
* person re-identification
* biometric profiling
* surveillance applications
* inferring sensitive personal attributes
* safety-critical deployment without additional validation and safeguards

## Limitations

Despite its scale and richness, Xperience-10M still has limitations.

* It reflects the environments, devices, and activity distributions represented in the collected data.
* Depth, pose, SLAM, and mocap annotations may contain noise or estimation error.
* Semantic annotations may not fully capture every relevant contextual factor in an episode.
* The scale of the dataset may require substantial storage and compute infrastructure for training.

## Social Impact

Xperience-10M can help advance world models, embodied AI, assistive systems, spatial intelligence, and robot learning from real-world human experience.

At the same time, egocentric multimodal data raises important questions around privacy, consent, and downstream misuse. We encourage all users to work with the dataset responsibly and to align usage with privacy protection, human-centered AI principles, and beneficial real-world applications.

## Privacy, Ethics, and Consent

Because Xperience-10M contains egocentric recordings of real-world human activity, privacy and consent are central considerations.

All data in Xperience-10M was collected and processed under appropriate consent and review procedures. Personally identifying or sensitive content is handled according to the dataset release policy. Access to some or all portions of the dataset may be controlled to protect participant privacy and support responsible use.

## Access

Xperience-10M is released for **research and other non-commercial uses**.

Because of the scale of the dataset (**~1 PB**) and the sensitive nature of egocentric multimodal data, access may be provided through controlled distribution channels. Users are expected to follow the dataset usage terms and any accompanying privacy, security, or redistribution requirements released with the dataset.

Before using Xperience-10M, please make sure you understand:
- the non-commercial restriction
- attribution requirements
- any privacy and responsible-use conditions associated with the data
- any additional access procedures specified by the dataset maintainers

## Citation

```bibtex
@dataset{xperience_10m,
  title={Xperience-10M: A Large-Scale Egocentric Multimodal Dataset with Structured 3D/4D Annotations},
  author={Ropedia},
  year={2026},
  publisher={Hugging Face},
  note={Dataset}
}
```
