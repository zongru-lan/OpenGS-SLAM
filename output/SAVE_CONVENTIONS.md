# OpenGS-SLAM Railway Clean Export Conventions

Last updated: 2026-06-07

This document records the agreed output convention for running OpenGS-SLAM on the seven railway monocular sequences under `/home/leizongru/lzr_ws/railway_data`.

## Core Principle

Do not rewrite the algorithm pipeline to fit our reporting needs.

For each SLAM/VO project, keep the project's internal files needed for runtime, debugging, and native evaluation. After the project reaches its final state, append a small dataset-specific clean export layer that converts the final state into the user-facing result format.

For OpenGS-SLAM this means:

- Frontend tracking, backend mapping, keyframe selection, color refinement, and Gaussian map optimization remain unchanged.
- OpenGS-SLAM internal outputs are kept under `output/<sequence_name>/internal_runs/<timestamp>/`.
- The railway clean export layer runs after the final Gaussian map is available.
- User-facing results are rewritten under `output/<sequence_name>/renders`, `poses`, and `metrics`.
- Railway runs are quality-first: when `color_refinement` is enabled, clean export requires the color-refined final map. If color refinement fails, times out, or the backend dies, the sequence is treated as failed instead of exporting a degraded fallback; the batch runner records `FAIL` for that sequence and continues with the remaining queue.

## Output Root

All railway outputs are saved under:

```text
/home/leizongru/lzr_ws/OpenGS-SLAM/output/
```

Each sequence has one user-facing directory:

```text
/home/leizongru/lzr_ws/OpenGS-SLAM/output/<sequence_name>/
```

Example:

```text
/home/leizongru/lzr_ws/OpenGS-SLAM/output/scene_14_train/
```

The batch runner removes `renders/`, `poses/`, `metrics/`, and `config.yaml` at the start of each sequence run. A successful clean export then rewrites those user-facing files. If a sequence fails, no stale user-facing result should remain for that sequence. Internal OpenGS-SLAM run folders are preserved under `internal_runs/`.

## Expected Layout

```text
output/<sequence_name>/
├── config.yaml
├── renders/
│   └── <gt_frame_id>.png
├── poses/
│   ├── <gt_frame_id>.txt
│   ├── estimated_c2w.csv
│   ├── estimated_c2w_tum.txt
│   └── README.md
├── metrics/
│   ├── render_metrics.csv
│   ├── render_metrics_summary.json
│   ├── ate_metrics.json
│   └── ate_aligned_trajectory.csv
└── internal_runs/
    └── <timestamp>/
```

## Render Images

Only OpenGS-SLAM selected keyframes are exported.

After the final Gaussian map is available, the clean export rerenders every saved keyframe pose using that final map. Only these final-map rerenders are exposed under `renders/`.

The rendered RGB is produced at the OpenGS-SLAM railway working resolution and then upsampled to the original railway RGB resolution:

```text
height = 2504
width  = 4112
```

The filename is the GT image prefix before the first underscore.

Example:

```text
GT image:
/home/leizongru/lzr_ws/railway_data/scene_05_train/048_1637935055.000000000.png

Saved render:
/home/leizongru/lzr_ws/OpenGS-SLAM/output/scene_05_train/renders/048.png
```

When `Dataset.undistort: true`, render metrics compare against the undistorted GT RGB image, because that is the image domain OpenGS-SLAM actually used.

## Pose Export

Poses are saved under:

```text
output/<sequence_name>/poses/
```

Each keyframe has one pose matrix:

```text
poses/<gt_frame_id>.txt
```

The pose convention is:

```text
T_world_camera / c2w
```

This matrix transforms homogeneous points from the camera frame to the OpenGS-SLAM internal map frame.

Coordinate-frame note:

```text
camera frame: +x right, +y down, +z forward
world/map frame: initialized internally by OpenGS-SLAM; not GPS, ENU, latitude/longitude, or the raw railway global pose frame
```

`poses/estimated_c2w.csv` includes explicit GT association fields:

```text
dataset_index
gt_frame_index
gt_frame_id
gt_image_name
gt_image_path
gt_timestamp
gt_pose_index
gt_pose_time_error_sec
render_name
pose_file
tx ty tz
qx qy qz qw
m00 ... m33
```

Use `gt_timestamp`, `gt_image_name`, and `gt_pose_index` to match exported poses back to `/home/leizongru/lzr_ws/railway_data/gt_poses/<sequence_name>.parquet`.

`poses/estimated_c2w_tum.txt` uses:

```text
gt_timestamp tx ty tz qx qy qz qw
```

## Rendering Metrics

Rendering metrics are saved under:

```text
output/<sequence_name>/metrics/
```

Files:

```text
render_metrics.csv
render_metrics_summary.json
```

Metrics:

```text
PSNR: computed on the saved GT-resolution RGB render
SSIM: computed on the saved GT-resolution RGB render
LPIPS: computed after resizing both render and GT to width 1024 while preserving aspect ratio
```

## ATE Metrics

ATE files:

```text
ate_metrics.json
ate_aligned_trajectory.csv
```

ATE convention:

```text
estimated trajectory: exported keyframe T_world_camera / c2w poses in the OpenGS-SLAM internal map frame
GT trajectory: railway parquet poses converted by RailwayDataset to the same first-frame-normalized GT frame used during loading
matching key: each exported keyframe carries its dataset index, GT image timestamp, and parquet pose index
primary metric: Sim(3) aligned ATE RMSE
```

`ate_metrics.json` also stores raw and SE(3)-aligned ATE for reference. For monocular VO, use `sim3_aligned.rmse_m` as the primary value.

## Notes

- Clean export is a reporting layer, not part of the algorithm.
- The final render images are final-map rerenders, not online intermediate renders.
- The exported render names and pose names always map back to the corresponding GT image prefix.
- Internal OpenGS-SLAM artifacts remain available under `internal_runs/` but should not be treated as the final user-facing result set.
