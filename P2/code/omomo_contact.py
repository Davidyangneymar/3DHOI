"""Extract OMOMO hand/object proximity and create review visualizations.

Uses the official 24-joint evaluation convention: hands are joints 22/23 and
contact is nearest object vertex distance < 0.05 m. These are estimated labels,
not measured physical contacts. Raw full sequences are the default input;
preprocessed windows remain available for comparisons.

Example:
    python omomo_contact.py --sequence sub16_plasticbox_000
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from plot_omomo_alignment import DATA_ROOT, PARENTS, read_obj_vertices


def global_joints_from_raw(record: dict) -> np.ndarray:
    """Reproduce OMOMO's 24-joint forward kinematics in the original world frame."""
    root = np.asarray(record["root_orient"], dtype=np.float64)
    body = np.asarray(record["pose_body"], dtype=np.float64).reshape(-1, 21, 3)
    local_rotvec = np.concatenate([root[:, None], body], axis=1)
    local_rotation = Rotation.from_rotvec(local_rotvec.reshape(-1, 3)).as_matrix()
    local_rotation = local_rotation.reshape(len(root), 22, 3, 3)
    offsets = np.asarray(record["rest_offsets"], dtype=np.float64)
    if offsets.shape != (24, 3):
        raise ValueError(f"Expected 24 rest offsets, found {offsets.shape}")
    positions = np.empty((len(root), 24, 3), dtype=np.float64)
    global_rotation = np.empty((len(root), 22, 3, 3), dtype=np.float64)
    positions[:, 0] = record["trans"] - record["trans2joint"]
    global_rotation[:, 0] = local_rotation[:, 0]
    for joint in range(1, 24):
        parent = PARENTS[joint]
        positions[:, joint] = positions[:, parent] + np.einsum(
            "tij,j->ti", global_rotation[:, parent], offsets[joint]
        )
        if joint < 22:
            global_rotation[:, joint] = global_rotation[:, parent] @ local_rotation[:, joint]
    return positions


def raw_record_as_window(record: dict) -> dict:
    """Adapt the raw sequence to the viewer's common field names."""
    joints = global_joints_from_raw(record)
    adapted = {
        "seq_name": record["seq_name"],
        "motion": joints.reshape(len(joints), 72),
        "obj_scale": record["obj_scale"],
        "obj_trans": record["obj_trans"][:, :, 0],
        "obj_rot_mat": record["obj_rot"],
    }
    if "obj_bottom_trans" in record:
        adapted.update({
            "obj_bottom_scale": record["obj_bottom_scale"],
            "obj_bottom_trans": record["obj_bottom_trans"][:, :, 0],
            "obj_bottom_rot_mat": record["obj_bottom_rot"],
        })
    return adapted


@lru_cache(maxsize=20)
def object_parts(name: str) -> list[tuple[str, np.ndarray, str]]:
    root = DATA_ROOT / "captured_objects"
    if name in {"mop", "vacuum"}:
        return [
            ("top", read_obj_vertices(root / f"{name}_cleaned_simplified_top.obj"), "obj"),
            ("bottom", read_obj_vertices(root / f"{name}_cleaned_simplified_bottom.obj"), "obj_bottom"),
        ]
    return [("whole", read_obj_vertices(root / f"{name}_cleaned_simplified.obj"), "obj")]


@lru_cache(maxsize=20)
def object_part_trees(name: str):
    return [(label, vertices, prefix, cKDTree(vertices))
            for label, vertices, prefix in object_parts(name)]


def extract_distances(window: dict, joints: np.ndarray, name: str):
    """Query static local meshes; rigid transforms preserve nearest vertices."""
    hands = joints[:, [22, 23]]
    count = len(hands)
    distances = np.full((count, 2), np.inf, dtype=np.float64)
    nearest = np.empty((count, 2, 3), dtype=np.float64)
    nearest_part = np.full((count, 2), -1, dtype=np.int32)
    for part_id, (_, vertices, prefix, tree) in enumerate(object_part_trees(name)):
        rotation_key = "obj_rot_mat" if prefix == "obj" else "obj_bottom_rot_mat"
        rotation = np.asarray(window[rotation_key], dtype=np.float64)
        scale = np.asarray(window[f"{prefix}_scale"], dtype=np.float64)
        translation = np.asarray(window[f"{prefix}_trans"], dtype=np.float64)
        local_distance = np.empty((count, 2), dtype=np.float64)
        local_index = np.zeros((count, 2), dtype=np.int64)
        nonzero = np.abs(scale) > 1e-12
        if np.any(nonzero):
            local_hands = np.einsum(
                "thi,tij->thj", hands[nonzero] - translation[nonzero, None], rotation[nonzero]
            ) / scale[nonzero, None, None]
            d, i = tree.query(local_hands.reshape(-1, 3))
            local_distance[nonzero] = d.reshape(-1, 2)
            local_index[nonzero] = i.reshape(-1, 2)
        if np.any(~nonzero):
            # A zero scale collapses every object vertex to the translation.
            local_distance[~nonzero] = np.linalg.norm(
                hands[~nonzero] - translation[~nonzero, None], axis=-1
            )
        part_distance = np.abs(scale[:, None]) * local_distance
        part_distance[~nonzero] = local_distance[~nonzero]
        local_nearest = vertices[local_index]
        part_nearest = scale[:, None, None] * np.einsum(
            "thi,tji->thj", local_nearest, rotation
        ) + translation[:, None]
        take = part_distance < distances
        distances[take] = part_distance[take]
        nearest[take] = part_nearest[take]
        nearest_part[take] = part_id
    return distances, nearest, nearest_part


def world_vertices(window: dict, parts: list[tuple[str, np.ndarray, str]], frame: int):
    world = []
    part_ids = []
    for part_id, (_, vertices, prefix) in enumerate(parts):
        rotation_key = "obj_rot_mat" if prefix == "obj" else "obj_bottom_rot_mat"
        transformed = (
            window[f"{prefix}_scale"][frame] * (vertices @ window[rotation_key][frame].T)
            + window[f"{prefix}_trans"][frame]
        )
        world.append(transformed)
        part_ids.extend([part_id] * len(transformed))
    return np.concatenate(world), np.asarray(part_ids)


def skeleton_trace(joints: np.ndarray) -> go.Scatter3d:
    lines = []
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            lines.extend([joints[parent], joints[joint], [np.nan, np.nan, np.nan]])
    lines = np.asarray(lines)
    return go.Scatter3d(
        x=lines[:, 0], y=lines[:, 1], z=lines[:, 2], mode="lines",
        line=dict(color="#243447", width=5), name="human skeleton",
    )


def frame_traces(window, parts, joints, frame, distances, nearest, threshold):
    points, _ = world_vertices(window, parts, frame)
    sampled = points[:: max(1, len(points) // 1200)]
    hands = joints[frame, [22, 23]]
    status = distances[frame] < threshold
    traces = [
        go.Scatter3d(
            x=sampled[:, 0], y=sampled[:, 1], z=sampled[:, 2],
            mode="markers", marker=dict(size=2, color="#669bbc", opacity=0.45),
            name="object vertices",
        ),
        skeleton_trace(joints[frame]),
    ]
    for index, color, label in [(0, "#e76f51", "left"), (1, "#3a86ff", "right")]:
        hand = hands[index]
        target = nearest[frame, index]
        traces.append(go.Scatter3d(
            x=[hand[0]], y=[hand[1]], z=[hand[2]], mode="markers",
            marker=dict(size=9 if status[index] else 6, color=color),
            name=f"{label} hand",
        ))
        traces.append(go.Scatter3d(
            x=[hand[0], target[0]], y=[hand[1], target[1]], z=[hand[2], target[2]],
            mode="lines+markers", line=dict(color=color, width=4),
            marker=dict(size=3, color=color), name=f"{label} nearest vertex",
        ))
    return traces


def save_timeline(path: Path, distances: np.ndarray, threshold: float, source: str):
    frames = np.arange(len(distances))
    fig, ax = plt.subplots(figsize=(12, 4))
    for index, color, label in [(0, "#e76f51", "left"), (1, "#3a86ff", "right")]:
        ax.plot(frames, distances[:, index], color=color, label=f"{label} hand")
        ax.fill_between(frames, 0, threshold, where=distances[:, index] < threshold,
                        color=color, alpha=0.20)
    ax.axhline(threshold, color="black", ls="--", lw=1, label=f"threshold {threshold:.2f} m")
    ax.set(xlabel="Sequence frame" if source == "raw" else "Window frame",
           ylabel="Nearest-vertex distance (m)")
    ax.set_xlim(0, len(frames) - 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_viewer(path: Path, window: dict, parts, joints, distances, nearest, threshold):
    sampled_frames = sorted(set(range(0, len(joints), 4)) | {len(joints) - 1})
    initial = sampled_frames[0]
    frames = [go.Frame(
        data=frame_traces(window, parts, joints, frame, distances, nearest, threshold),
        name=str(frame),
        layout=go.Layout(title_text=(
            f"{window['seq_name']} | frame {frame} | "
            f"L {distances[frame, 0]:.3f} m / R {distances[frame, 1]:.3f} m"
        )),
    ) for frame in sampled_frames]
    fig = go.Figure(data=frame_traces(window, parts, joints, initial, distances, nearest, threshold), frames=frames)
    all_points = [joints.reshape(-1, 3)]
    for frame in sampled_frames:
        points, _ = world_vertices(window, parts, frame)
        all_points.append(points[:: max(1, len(points) // 500)])
    cloud = np.concatenate(all_points)
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    center = (low + high) / 2
    half = max((high - low).max() * 0.58, 0.5)
    ranges = [[float(c - half), float(c + half)] for c in center]
    fig.update_layout(
        title=f"{window['seq_name']} | frame {initial} | threshold {threshold:.2f} m",
        scene=dict(
            xaxis=dict(title="X (m)", range=ranges[0]),
            yaxis=dict(title="Y (m)", range=ranges[1]),
            zaxis=dict(title="Z (m)", range=ranges[2]),
            aspectmode="cube",
        ),
        sliders=[dict(
            steps=[dict(method="animate", label=str(frame), args=[
                [str(frame)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                               "transition": {"duration": 0}},
            ]) for frame in sampled_frames],
            currentvalue=dict(prefix="Frame: "),
        )],
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 120, "redraw": True},
                                                        "transition": {"duration": 0}}]),
            dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0}}]),
        ])],
        margin=dict(l=0, r=0, t=55, b=0),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="sub16_plasticbox_000")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--source", choices=["raw", "window"], default="raw")
    parser.add_argument("--start", type=int, default=None, help="Window start frame; used only with --source window")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contact"))
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1 metre")
    if args.source == "raw":
        records = joblib.load(DATA_ROOT / f"{args.split}_diffusion_manip_seq_joints24.p")
        record = next((value for value in records.values() if value["seq_name"] == args.sequence), None)
        if record is None:
            parser.error(f"Sequence not found in {args.split} raw data")
        window = raw_record_as_window(record)
        start = 0
        suffix = "full"
    else:
        if args.split != "test":
            parser.error("--source window currently supports --split test only")
        windows = joblib.load(DATA_ROOT / "test_diffusion_manip_window_120_processed_joints24.p")
        matches = [value for value in windows.values() if value["seq_name"] == args.sequence]
        window = next((value for value in matches if value["start_t_idx"] == args.start), None) if args.start is not None else (
            min(matches, key=lambda value: value["start_t_idx"]) if matches else None
        )
        if window is None:
            parser.error("Sequence/window not found in test window data")
        start = int(window["start_t_idx"])
        suffix = f"start{start}"
    joints = window["motion"][:, :72].reshape(-1, 24, 3)
    parts = object_parts(args.sequence.split("_")[1])
    distances, nearest, nearest_part = extract_distances(window, joints, args.sequence.split("_")[1])

    flags = distances < args.threshold
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sequence}_{suffix}"
    csv_path = args.output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "source_frame", "window_frame", "left_distance_m", "right_distance_m",
                         "left_contact_est", "right_contact_est", "left_part", "right_part",
                         "left_nearest_x", "left_nearest_y", "left_nearest_z",
                         "right_nearest_x", "right_nearest_y", "right_nearest_z"])
        for frame in range(len(joints)):
            writer.writerow([args.sequence, start + frame, frame, *distances[frame],
                             *[int(x) for x in flags[frame]],
                             *[parts[index][0] for index in nearest_part[frame]],
                             *nearest[frame, 0], *nearest[frame, 1]])
    np.savez_compressed(args.output_dir / f"{stem}.npz", distances_m=distances,
                        contact_est=flags, nearest_vertex_xyz=nearest,
                        nearest_part=nearest_part, threshold_m=args.threshold,
                        source_start_frame=start)
    save_timeline(args.output_dir / f"{stem}_timeline.png", distances, args.threshold, args.source)
    save_viewer(args.output_dir / f"{stem}_viewer.html", window, parts, joints, distances, nearest, args.threshold)
    print(f"sequence={args.sequence}; frames={len(joints)}; parts={[p[0] for p in parts]}")
    print(f"estimated contact frames: left={int(flags[:, 0].sum())}, right={int(flags[:, 1].sum())}")
    print(f"outputs: {args.output_dir.resolve()} / {stem}.*")


if __name__ == "__main__":
    main()
