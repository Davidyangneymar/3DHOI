"""Create a reproducible type x severity review sheet and local 3D viewers."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from counterfactual.build_v1 import OUT, THRESHOLD, TYPES
from omomo_contact import object_parts, save_viewer, world_vertices


def main() -> None:
    rows = [json.loads(x) for x in (OUT / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen = []
    used_sequences = set()
    used_objects = set()
    for kind in TYPES:
        for severity in ("low", "medium", "high"):
            candidates = [e for e in rows if e["split"] == "test" and e["type"] == kind and e["severity"] == severity]
            if candidates:
                pick = min(candidates, key=lambda e: (e["source_sequence"] in used_sequences,
                                                      e["object"] in used_objects, e["sample_id"]))
                chosen.append(pick)
                used_sequences.add(pick["source_sequence"])
                used_objects.add(pick["object"])
    output = OUT / "review"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharey=False)
    links = []
    for ax, entry in zip(axes.flat, chosen):
        path = OUT / "samples" / f"{entry['sample_id']}.npz"
        with np.load(path) as z:
            sample = {k: z[k] for k in z.files}
        name = entry["source_sequence"]
        clean_path = OUT / "samples" / f"{name}__clean__none.npz"
        with np.load(clean_path) as z:
            clean_dist = z["distances_m"]
        dist = sample["distances_m"]
        frames = np.arange(len(dist))
        ax.plot(frames, clean_dist[:, 0], "--", c="#e76f51", alpha=.45, label="L clean")
        ax.plot(frames, clean_dist[:, 1], "--", c="#3a86ff", alpha=.45, label="R clean")
        ax.plot(frames, dist[:, 0], c="#e76f51", label="L corrupted")
        ax.plot(frames, dist[:, 1], c="#3a86ff", label="R corrupted")
        ax.axhline(THRESHOLD, c="black", ls=":", lw=1)
        ax.set_title(f"{entry['type']} / {entry['severity']}\n{name}", fontsize=9)
        ax.set_xlabel("frame")
        ax.set_ylabel("hand-object distance (m)")
        ax.grid(alpha=.2)
        window = {"seq_name": entry["sample_id"], "motion": sample["joints"].reshape(len(dist), 72),
                  "obj_trans": sample["obj_trans"], "obj_rot_mat": sample["obj_rot_mat"],
                  "obj_scale": sample["obj_scale"]}
        for key in ("obj_bottom_trans", "obj_bottom_rot_mat", "obj_bottom_scale"):
            if key in sample:
                window[key] = sample[key]
        html = output / f"{entry['sample_id']}.html"
        save_viewer(html, window, object_parts(entry["object"]), sample["joints"],
                    dist, sample["nearest_xyz"], THRESHOLD)
        links.append((entry, html.name))
        print(f"viewer {len(links)}/{len(chosen)} {html.name}", flush=True)
    fig.tight_layout()
    fig.savefig(output / "stratified_contact_timeline.png", dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(16, 13))
    for i, entry in enumerate(chosen):
        with np.load(OUT / "samples" / f"{entry['sample_id']}.npz") as z:
            current = {k: z[k] for k in z.files}
        with np.load(OUT / "samples" / f"{entry['source_sequence']}__clean__none.npz") as z:
            original = {k: z[k] for k in z.files}
        frame = min(len(current["joints"]) - 1, int(len(current["joints"]) * .75))
        def as_window(sample):
            w = {"obj_trans": sample["obj_trans"], "obj_rot_mat": sample["obj_rot_mat"], "obj_scale": sample["obj_scale"]}
            for key in ("obj_bottom_trans", "obj_bottom_rot_mat", "obj_bottom_scale"):
                if key in sample:
                    w[key] = sample[key]
            return w
        clean_obj, _ = world_vertices(as_window(original), object_parts(entry["object"]), frame)
        corrupt_obj, _ = world_vertices(as_window(current), object_parts(entry["object"]), frame)
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        ax.scatter(*clean_obj[::max(1, len(clean_obj) // 350)].T, s=1, c="#aaaaaa", alpha=.25)
        ax.scatter(*corrupt_obj[::max(1, len(corrupt_obj) // 350)].T, s=1, c="#008b8b", alpha=.25)
        before = original["joints"][frame, [22, 23]]
        after = current["joints"][frame, [22, 23]]
        for h, color in ((0, "#d9482b"), (1, "#176bd1")):
            ax.scatter(*before[h], s=40, facecolors="none", edgecolors=color, linewidths=2)
            ax.scatter(*after[h], s=38, c=color)
            ax.plot([before[h, 0], after[h, 0]], [before[h, 1], after[h, 1]],
                    [before[h, 2], after[h, 2]], c=color, lw=2)
        cloud = np.vstack((clean_obj, corrupt_obj, before, after))
        center = cloud.mean(axis=0)
        radius = max(np.ptp(cloud, axis=0).max() * .55, .18)
        ax.set(xlim=(center[0]-radius, center[0]+radius),
               ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.view_init(elev=25, azim=40)
        ax.set_title(f"{entry['type']} / {entry['severity']}\n{entry['source_sequence']} frame {frame}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("Grey: clean object; teal: corrupted object; open circles: clean hands; filled: corrupted hands", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "stratified_3d_snapshot.png", dpi=150)
    plt.close(fig)
    table = "\n".join(f"<tr><td>{e['type']}</td><td>{e['severity']}</td><td>{e['source_sequence']}</td><td><a href='{p}'>3D viewer</a></td></tr>" for e, p in links)
    (output / "index.html").write_text("<!doctype html><html><meta charset='utf-8'><title>Counterfactual v1 review</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;color:#243447}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>"
        "<h1>Counterfactual Dataset v1 · stratified review</h1><p>Each row is a held-out sequence. Dashed lines show the clean hand distances. "
        "Solid lines show the corruption; dotted line is the 0.05 m contact threshold. Open a 3D viewer and use its frame slider.</p>"
        "<img src='stratified_contact_timeline.png'><img src='stratified_3d_snapshot.png'>"
        "<table><tr><th>type</th><th>severity</th><th>sequence</th><th>viewer</th></tr>"
        + table + "</table></html>", encoding="utf-8")
    (output / "selection.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
