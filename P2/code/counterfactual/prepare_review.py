"""Build an offline, score-free review queue without changing the frozen dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs
from models.interaction_critic import ROOT
from counterfactual.critic_api import sha256
from omomo_contact import object_parts
from plot_omomo_alignment import PARENTS

OUT = ROOT / "counterfactual" / "v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refusing to overwrite reviews")
    entries = [json.loads(line) for line in (OUT / "index.jsonl").read_text().splitlines()]
    texts = {r["sequence"]: r["description"] for r in csv.DictReader((ROOT / "outputs/text_audit/sequence_text.csv").open(encoding="utf-8-sig"))}
    second = {e["sample_id"] for e in entries if e["type"] == "spatial_contact_shift"}
    strata = {}
    for e in entries:
        strata.setdefault((e["type"], e["severity"]), []).append(e["sample_id"])
    for names in strata.values():
        ordered = sorted(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
        second.update(ordered[:math.ceil(len(names) * .2)])
    meshes = {name: [{"prefix": prefix, "vertices": vertices[::max(1, math.ceil(len(vertices) / 700))].round(5).tolist()}
                      for _, vertices, prefix in object_parts(name)] for name in sorted({e["object"] for e in entries})}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "samples").mkdir()
    for i, e in enumerate(entries, 1):
        path = OUT / "samples" / f"{e['sample_id']}.npz"
        if sha256(path) != e["sha256"]:
            raise ValueError(f"Frozen sample changed: {path}")
        with np.load(path, allow_pickle=False) as z:
            sample = {k: z[k].round(5).tolist() for k in z.files if k in {"joints", "distances_m"} or k.startswith("obj_")}
        (args.output / "samples" / f"{e['sample_id']}.js").write_text(
            f"window.samples[{json.dumps(e['sample_id'])}]={json.dumps(sample, separators=(',', ':'))};", encoding="utf-8")
        e.update({"description": texts.get(e["source_sequence"], ""), "second_review_required": e["sample_id"] in second})
        if i % 100 == 0:
            print(f"Prepared {i}/{len(entries)}", flush=True)
    payload = {"entries": entries, "meshes": meshes, "parents": [int(x) for x in PARENTS]}
    (args.output / "queue.js").write_text("window.queue=" + json.dumps(payload, ensure_ascii=True) + ";", encoding="utf-8")
    (args.output / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    (args.output / "index.html").write_text((Path(__file__).with_name("review_template.html")).read_text(encoding="utf-8"), encoding="utf-8")
    for asset in ("review_form.js", "review_app.js"):
        shutil.copy2(Path(__file__).with_name(asset), args.output / asset)
    (args.output / "queue.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    (args.output / "provenance.json").write_text(json.dumps({"status": "awaiting_human_review",
        "sample_count": len(entries), "sources": len({e["source_sequence"] for e in entries}),
        "second_review_required": len(second), "manifest_sha256": sha256(OUT / "manifest.json"),
        "builder_sha256": sha256(Path(__file__)), "labels_approved": 0}, indent=2), encoding="utf-8")
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
