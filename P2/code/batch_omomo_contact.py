"""Extract estimated hand/object contact for every raw OMOMO sequence.

Example:
    python batch_omomo_contact.py --split test

Writes one compact NPZ per sequence and a CSV summary. Review flags identify
sequences to inspect; they are not ground-truth error labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from omomo_contact import DATA_ROOT, extract_distances, raw_record_as_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1 metre")
    output_dir = args.output_dir or Path("outputs/contact_batch") / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    records = joblib.load(DATA_ROOT / f"{args.split}_diffusion_manip_seq_joints24.p")
    summary = []
    errors = []
    for index, record in enumerate(records.values(), start=1):
        name = record["seq_name"]
        object_name = name.split("_")[1]
        try:
            adapted = raw_record_as_window(record)
            joints = adapted["motion"].reshape(-1, 24, 3)
            distances, nearest, part_ids = extract_distances(adapted, joints, object_name)
            if not np.isfinite(distances).all() or not np.isfinite(nearest).all():
                raise ValueError("non-finite contact output")
            flags = distances < args.threshold
            scales = [record["obj_scale"]]
            if "obj_bottom_scale" in record:
                scales.append(record["obj_bottom_scale"])
            zero_scale_frames = int(np.count_nonzero(np.any(
                np.stack([np.abs(scale) < 1e-12 for scale in scales]), axis=0
            )))
            review_flags = []
            if zero_scale_frames:
                review_flags.append("zero_scale")
            if not flags.any():
                review_flags.append("no_contact")
            if flags.all():
                review_flags.append("both_hands_always_contact")
            np.savez_compressed(
                output_dir / f"{name}.npz",
                sequence=name, distances_m=distances, contact_est=flags,
                nearest_vertex_xyz=nearest, nearest_part=part_ids,
                threshold_m=args.threshold, source_frame=np.arange(len(joints)),
            )
            summary.append({
                "sequence": name,
                "object": object_name,
                "frames": len(joints),
                "left_contact_frames": int(flags[:, 0].sum()),
                "right_contact_frames": int(flags[:, 1].sum()),
                "either_contact_frames": int(flags.any(axis=1).sum()),
                "left_min_distance_m": float(distances[:, 0].min()),
                "right_min_distance_m": float(distances[:, 1].min()),
                "zero_scale_frames": zero_scale_frames,
                "review_flags": ";".join(review_flags),
            })
        except Exception as exc:
            errors.append((name, type(exc).__name__, str(exc)))
        if index % 100 == 0:
            print(f"processed {index}/{len(records)}; errors={len(errors)}", flush=True)

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "sequence", "object", "frames", "left_contact_frames",
            "right_contact_frames", "either_contact_frames", "left_min_distance_m",
            "right_min_distance_m", "zero_scale_frames", "review_flags",
        ])
        writer.writeheader()
        writer.writerows(summary)
    errors_path = output_dir / "errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "error_type", "message"])
        writer.writerows(errors)
    print(f"completed={len(summary)} failed={len(errors)} output={output_dir.resolve()}")


if __name__ == "__main__":
    main()
