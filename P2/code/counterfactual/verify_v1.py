"""Verify the frozen dataset manifest and per-sample hashes."""
from __future__ import annotations

import json
from collections import Counter

from counterfactual.build_v1 import OUT, ROOT, sha256


def main() -> None:
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    entries = [json.loads(line) for line in (OUT / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert m["dataset"] == "Counterfactual Dataset v1"
    assert m["sample_count"] == len(entries)
    assert m["index_sha256"] == sha256(OUT / "index.jsonl")
    assert m["review_sha256"] == sha256(OUT / "review_labels.csv")
    assert m["builder_sha256"] == sha256(ROOT / "counterfactual" / "build_v1.py")
    for source, digest in m["raw_source_sha256"].items():
        assert digest == sha256(ROOT / "data" / f"{source}_diffusion_manip_seq_joints24.p")
    seen = set()
    groups = {}
    for e in entries:
        assert e["sample_id"] not in seen
        seen.add(e["sample_id"])
        groups.setdefault(e["source_sequence"], set()).add(e["split"])
        assert e["sha256"] == sha256(OUT / "samples" / f"{e['sample_id']}.npz"), e["sample_id"]
    assert all(len(x) == 1 for x in groups.values())
    assert {p.stem for p in (OUT / "samples").glob("*.npz")} == seen
    assert dict(Counter(e["split"] for e in entries)) == m["by_split"]
    print(f"verified {len(entries)} samples from {len(groups)} sequence-disjoint sources")


if __name__ == "__main__":
    main()
