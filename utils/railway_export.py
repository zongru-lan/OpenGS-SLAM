import csv
import json
import os
import shutil
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from scipy.spatial.transform import Rotation as R
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian_splatting.gaussian_renderer import render
from utils.logging_utils import Log


def export_railway_results(
    frames,
    kf_indices,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    config,
):
    if config["Dataset"].get("type") != "railway":
        return
    if not kf_indices:
        raise RuntimeError("Railway clean export requires at least one keyframe.")

    sequence_dir = _sequence_output_dir(config, save_dir)
    renders_dir = os.path.join(sequence_dir, "renders")
    poses_dir = os.path.join(sequence_dir, "poses")
    metrics_dir = os.path.join(sequence_dir, "metrics")

    for path in (renders_dir, poses_dir, metrics_dir):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(sequence_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    lpips_metric.eval()

    pose_rows = []
    render_rows = []
    est_poses = []
    gt_poses = []
    records = []

    with torch.no_grad():
        for frame_idx in kf_indices:
            frame = frames[frame_idx]
            frame_info = _frame_info(dataset, frame_idx)
            render_name = f"{frame_info['gt_frame_id']}.png"
            pose_name = f"{frame_info['gt_frame_id']}.txt"

            render_pkg = render(frame, gaussians, pipe, background)
            if render_pkg is None:
                raise RuntimeError(f"Render returned None for keyframe {frame_idx}")
            render_rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
            render_np = _tensor_chw_to_uint8(render_rgb)

            gt_np = _load_gt_rgb_for_metrics(dataset, frame_idx)
            render_full = _resize_rgb(render_np, gt_np.shape[1], gt_np.shape[0])
            Image.fromarray(render_full).save(os.path.join(renders_dir, render_name))

            psnr_value = _psnr_np(render_full, gt_np)
            ssim_value = _ssim_np(render_full, gt_np)
            lpips_value = _lpips_np(render_full, gt_np, lpips_metric)

            est_c2w = _camera_c2w(frame.R, frame.T)
            gt_c2w = _camera_c2w(frame.R_gt, frame.T_gt)
            np.savetxt(os.path.join(poses_dir, pose_name), est_c2w, fmt="%.10f")

            est_quat = R.from_matrix(est_c2w[:3, :3]).as_quat()
            pose_row = {
                **frame_info,
                "render_name": render_name,
                "pose_file": pose_name,
                "tx": float(est_c2w[0, 3]),
                "ty": float(est_c2w[1, 3]),
                "tz": float(est_c2w[2, 3]),
                "qx": float(est_quat[0]),
                "qy": float(est_quat[1]),
                "qz": float(est_quat[2]),
                "qw": float(est_quat[3]),
            }
            for r in range(4):
                for c in range(4):
                    pose_row[f"m{r}{c}"] = float(est_c2w[r, c])
            pose_rows.append(pose_row)

            render_rows.append({
                **frame_info,
                "render_name": render_name,
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
            })
            est_poses.append(est_c2w)
            gt_poses.append(gt_c2w)
            records.append({**frame_info, "render_name": render_name})

    _write_csv(os.path.join(poses_dir, "estimated_c2w.csv"), pose_rows)
    _write_tum(os.path.join(poses_dir, "estimated_c2w_tum.txt"), pose_rows)
    _write_pose_readme(os.path.join(poses_dir, "README.md"), config)

    _write_csv(os.path.join(metrics_dir, "render_metrics.csv"), render_rows)
    render_summary = _summarize_render_metrics(render_rows, config, len(kf_indices))
    with open(
        os.path.join(metrics_dir, "render_metrics_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(render_summary, f, indent=4)

    ate_summary, aligned_rows = _compute_ate(records, gt_poses, est_poses)
    with open(os.path.join(metrics_dir, "ate_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(ate_summary, f, indent=4)
    _write_csv(os.path.join(metrics_dir, "ate_aligned_trajectory.csv"), aligned_rows)

    Log("Railway clean export saved to " + sequence_dir, tag="Eval")


def _sequence_output_dir(config, save_dir):
    configured = config["Results"].get("sequence_save_dir")
    if configured:
        return configured
    if save_dir and os.path.basename(os.path.dirname(save_dir)) == "internal_runs":
        return os.path.dirname(os.path.dirname(save_dir))
    if save_dir:
        return os.path.dirname(save_dir)
    scene = config["Dataset"].get("scene")
    if not scene:
        scene = os.path.basename(config["Dataset"]["dataset_path"].rstrip("/"))
    return os.path.join(config["Results"]["save_dir"], scene)


def _frame_info(dataset, idx):
    frame_id = getattr(dataset, "frame_ids", [f"{idx:06d}"])[idx]
    image_name = getattr(dataset, "image_names", [os.path.basename(dataset.color_paths[idx])])[idx]
    timestamp = getattr(dataset, "image_timestamps", [float(idx)])[idx]
    gt_pose_indices = getattr(dataset, "gt_pose_indices", None)
    gt_pose_time_errors = getattr(dataset, "gt_pose_time_errors", None)
    return {
        "dataset_index": int(idx),
        "gt_frame_index": int(idx),
        "gt_frame_id": str(frame_id),
        "gt_image_name": image_name,
        "gt_image_path": dataset.color_paths[idx],
        "gt_timestamp": float(timestamp),
        "gt_pose_index": int(gt_pose_indices[idx]) if gt_pose_indices is not None else "",
        "gt_pose_time_error_sec": (
            float(gt_pose_time_errors[idx]) if gt_pose_time_errors is not None else ""
        ),
    }


def _load_gt_rgb_for_metrics(dataset, idx):
    image = np.array(Image.open(dataset.color_paths[idx]).convert("RGB"))
    if dataset.config["Dataset"].get("undistort", True):
        image = cv2.undistort(image, dataset.original_K, dataset.original_dist_coeffs)
    return image


def _resize_rgb(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return np.array(Image.fromarray(image).resize((width, height), resampling))


def _tensor_chw_to_uint8(image):
    return (
        image.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
        .clip(0.0, 1.0)
        * 255.0
    ).astype(np.uint8)


def _camera_c2w(R_tensor, T_tensor):
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_tensor.detach().cpu().numpy()
    w2c[:3, 3] = T_tensor.detach().cpu().numpy()
    return np.linalg.inv(w2c)


def _psnr_np(pred, gt):
    pred_f = pred.astype(np.float64) / 255.0
    gt_f = gt.astype(np.float64) / 255.0
    mse = np.mean((pred_f - gt_f) ** 2)
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _ssim_np(pred, gt):
    pred_f = pred.astype(np.float32) / 255.0
    gt_f = gt.astype(np.float32) / 255.0
    values = []
    for channel in range(3):
        x = pred_f[:, :, channel]
        y = gt_f[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        c1 = 0.01**2
        c2 = 0.03**2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        values.append(float(np.mean(ssim_map)))
    return float(np.mean(values))


def _lpips_np(pred, gt, lpips_metric):
    target_width = 1024
    if pred.shape[1] > target_width:
        target_height = int(round(pred.shape[0] * target_width / pred.shape[1]))
        pred = _resize_rgb(pred, target_width, target_height)
        gt = _resize_rgb(gt, target_width, target_height)
    pred_t = _image_uint8_to_tensor(pred).cuda()
    gt_t = _image_uint8_to_tensor(gt).cuda()
    value = lpips_metric(pred_t, gt_t)
    return float(value.detach().cpu().item())


def _image_uint8_to_tensor(image):
    return (
        torch.from_numpy(image.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_tum(path, pose_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# gt_timestamp tx ty tz qx qy qz qw\n")
        for row in pose_rows:
            f.write(
                "{gt_timestamp:.9f} {tx:.10f} {ty:.10f} {tz:.10f} "
                "{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}\n".format(**row)
            )


def _write_pose_readme(path, config):
    content = f"""# Railway Pose Export

Pose convention: `T_world_camera / c2w`.

The matrix transforms homogeneous points from the camera frame to the OpenGS-SLAM
internal map frame. The map frame is initialized by the monocular SLAM run; it is
not the railway global pose frame from parquet.

GT association is explicit in `estimated_c2w.csv` through `gt_frame_id`,
`gt_image_name`, `gt_timestamp`, and `gt_pose_index`.

Dataset scene: `{config['Dataset'].get('scene', '')}`
Generated at: `{datetime.now().isoformat(timespec='seconds')}`
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _summarize_render_metrics(rows, config, n_keyframes):
    keys = ["psnr", "ssim", "lpips"]
    summary = {
        "scene": config["Dataset"].get("scene"),
        "num_keyframes": n_keyframes,
        "render_convention": (
            "Each keyframe pose is rerendered once with the final Gaussian map. "
            "The low-resolution render is upsampled to the original railway RGB resolution."
        ),
        "metric_convention": {
            "psnr": "Computed on the saved GT-resolution RGB render.",
            "ssim": "Computed on the saved GT-resolution RGB render.",
            "lpips": "Computed after resizing both render and GT to width 1024.",
            "gt_image_preprocess": (
                "GT RGB is undistorted before metric calculation when Dataset.undistort is true."
            ),
        },
    }
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def _compute_ate(records, gt_poses, est_poses):
    traj_ref = PosePath3D(poses_se3=gt_poses)
    raw_traj = PosePath3D(poses_se3=est_poses)
    se3_traj = _align_trajectory_robust(gt_poses, est_poses, correct_scale=False)
    sim3_traj = _align_trajectory_robust(gt_poses, est_poses, correct_scale=True)

    raw_stats, raw_errors = _ape_stats(traj_ref, raw_traj)
    se3_stats, se3_errors = _ape_stats(traj_ref, se3_traj)
    sim3_stats, sim3_errors = _ape_stats(traj_ref, sim3_traj)

    aligned_rows = []
    for i, record in enumerate(records):
        gt_t = gt_poses[i][:3, 3]
        est_t = est_poses[i][:3, 3]
        se3_t = se3_traj.poses_se3[i][:3, 3]
        sim3_t = sim3_traj.poses_se3[i][:3, 3]
        aligned_rows.append({
            **record,
            "gt_x": float(gt_t[0]),
            "gt_y": float(gt_t[1]),
            "gt_z": float(gt_t[2]),
            "est_x": float(est_t[0]),
            "est_y": float(est_t[1]),
            "est_z": float(est_t[2]),
            "est_se3_aligned_x": float(se3_t[0]),
            "est_se3_aligned_y": float(se3_t[1]),
            "est_se3_aligned_z": float(se3_t[2]),
            "est_sim3_aligned_x": float(sim3_t[0]),
            "est_sim3_aligned_y": float(sim3_t[1]),
            "est_sim3_aligned_z": float(sim3_t[2]),
            "raw_error_m": float(raw_errors[i]),
            "se3_error_m": float(se3_errors[i]),
            "sim3_error_m": float(sim3_errors[i]),
        })

    summary = {
        "num_keyframes": len(records),
        "pose_convention": "T_world_camera / c2w",
        "estimated_frame": "OpenGS-SLAM internal monocular map frame.",
        "gt_frame": (
            "Railway parquet poses converted by RailwayDataset to a frame normalized "
            "by the first loaded image."
        ),
        "matching": (
            "One estimated keyframe pose is matched to the GT image and parquet pose "
            "record carried by RailwayDataset."
        ),
        "primary_metric": "sim3_aligned.rmse_m",
        "raw": _stats_for_json(raw_stats),
        "se3_aligned": _stats_for_json(se3_stats),
        "sim3_aligned": _stats_for_json(sim3_stats),
    }
    return summary, aligned_rows


def _ape_stats(traj_ref, traj_est):
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    return ape.get_all_statistics(), ape.error


def _align_trajectory_robust(gt_poses, est_poses, correct_scale):
    traj_ref = PosePath3D(poses_se3=gt_poses)
    try:
        return trajectory.align_trajectory(
            PosePath3D(poses_se3=est_poses), traj_ref, correct_scale=correct_scale
        )
    except Exception as exc:
        Log(
            f"evo alignment failed ({exc}); using point-based fallback alignment",
            tag="Eval",
        )
        return PosePath3D(
            poses_se3=_align_poses_by_translation(est_poses, gt_poses, correct_scale)
        )


def _align_poses_by_translation(est_poses, gt_poses, correct_scale):
    est_xyz = np.asarray([pose[:3, 3] for pose in est_poses], dtype=np.float64)
    gt_xyz = np.asarray([pose[:3, 3] for pose in gt_poses], dtype=np.float64)
    rot, trans, scale = _umeyama_points(est_xyz, gt_xyz, correct_scale)
    aligned = []
    for pose in est_poses:
        aligned_pose = np.array(pose, dtype=np.float64, copy=True)
        aligned_pose[:3, :3] = rot @ aligned_pose[:3, :3]
        aligned_pose[:3, 3] = scale * (rot @ aligned_pose[:3, 3]) + trans
        aligned.append(aligned_pose)
    return aligned


def _umeyama_points(source_xyz, target_xyz, with_scale):
    if len(source_xyz) != len(target_xyz):
        raise ValueError("source and target trajectories must have the same length")
    if len(source_xyz) == 0:
        raise ValueError("cannot align an empty trajectory")

    source_mean = source_xyz.mean(axis=0)
    target_mean = target_xyz.mean(axis=0)
    source_centered = source_xyz - source_mean
    target_centered = target_xyz - target_mean
    covariance = target_centered.T @ source_centered / len(source_xyz)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0:
        sign[-1] = -1.0
    rot = u_mat @ np.diag(sign) @ vt_mat

    scale = 1.0
    if with_scale:
        variance = np.mean(np.sum(source_centered**2, axis=1))
        if variance > 1e-12:
            scale = float(np.sum(singular_values * sign) / variance)
    trans = target_mean - scale * (rot @ source_mean)
    return rot, trans, scale


def _stats_for_json(stats):
    return {f"{key}_m": float(value) for key, value in stats.items()}
