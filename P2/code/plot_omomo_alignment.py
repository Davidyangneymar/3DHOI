"""Plot an OMOMO preprocessed window to inspect human/object alignment.

Usage:
    python plot_omomo_alignment.py --sequence sub16_plasticbox_000

The first 72 values in each `motion` frame are the 24 global joint positions
written by OMOMO's `cal_normalize_data_input`, before `__getitem__` normalizes
training inputs. Object vertices use the same transform as
`HandFootManipDataset.apply_transformation_to_obj_geometry`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


DATA_ROOT = Path(__file__).resolve().parent / "data"
PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
)


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"No vertices in {path}")
    return np.asarray(vertices, dtype=np.float64)


def transformed_object(window: dict, vertices: np.ndarray, frame: int) -> np.ndarray:
    rotation = window["obj_rot_mat"][frame]
    scale = window["obj_scale"][frame]
    translation = window["obj_trans"][frame]
    return scale * (vertices @ rotation.T) + translation


def set_equal_axes(ax, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    midpoint = (lower + upper) / 2
    radius = max((upper - lower).max() * 0.57, 0.25)
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    ax.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="sub16_plasticbox_000")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("omomo_alignment.png"))
    args = parser.parse_args()

    windows = joblib.load(DATA_ROOT / "test_diffusion_manip_window_120_processed_joints24.p")
    window = next((v for v in windows.values() if v["seq_name"] == args.sequence), None)
    if window is None:
        raise SystemExit(f"Sequence not found in test windows: {args.sequence}")
    name = args.sequence.split("_")[1]
    if name in {"mop", "vacuum"}:
        raise SystemExit("This first alignment plot supports single-part objects only.")
    mesh_path = DATA_ROOT / "captured_objects" / f"{name}_cleaned_simplified.obj"
    vertices = read_obj_vertices(mesh_path)
    joints = window["motion"][:, :72].reshape(-1, 24, 3)

    if args.frame is None:
        candidates = range(len(joints))
        distances = []
        for frame in candidates:
            object_points = transformed_object(window, vertices, frame)
            tree = cKDTree(object_points)
            distance, _ = tree.query(joints[frame, [22, 23]])
            distances.append(distance.min())
        frame = int(np.argmin(distances))
    else:
        frame = args.frame
    if not 0 <= frame < len(joints):
        raise SystemExit(f"Frame must be in 0..{len(joints) - 1}")

    body = joints[frame]
    object_points = transformed_object(window, vertices, frame)
    tree = cKDTree(object_points)
    hand_distance, nearest_idx = tree.query(body[[22, 23]])
    nearest = object_points[nearest_idx]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    stride = max(1, len(object_points) // 3500)
    sampled = object_points[::stride]
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=0.6, alpha=0.27, c="#567f9b")
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            segment = body[[joint, parent]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], c="#27374d", lw=2)
    ax.scatter(body[:, 0], body[:, 1], body[:, 2], s=15, c="#27374d")
    colors = ["#e76f51", "#3a86ff"]
    for index, label in enumerate(["left", "right"]):
        hand = body[22 + index]
        target = nearest[index]
        ax.scatter(*hand, s=75, c=colors[index], label=f"{label}: {hand_distance[index]:.3f} m")
        ax.scatter(*target, s=35, marker="x", c=colors[index])
        ax.plot(*np.stack([hand, target]).T, c=colors[index], ls="--", lw=1)
    set_equal_axes(ax, np.concatenate([body, object_points], axis=0))
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.set_title(f"{args.sequence} | window frame {frame} | vertex distances")
    ax.legend(loc="upper left")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f"saved: {args.output.resolve()}")
    print(f"frame: {frame}; left/right nearest-vertex distance (m): {hand_distance.tolist()}")


if __name__ == "__main__":
    main()
