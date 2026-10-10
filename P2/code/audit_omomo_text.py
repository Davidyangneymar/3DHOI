"""Join official OMOMO free-text annotations to the released sequence IDs.

Reads `omomo_text_anno.zip` in the project root and writes audit CSV files.
The descriptions are free text; this script does not invent action classes.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import joblib


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "text_audit"


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotations = {}
    issues = []
    with ZipFile(ROOT / "omomo_text_anno.zip") as archive:
        for member in archive.namelist():
            if not member.endswith(".json"):
                continue
            sequence = PurePosixPath(member).stem
            value = json.loads(archive.read(member))
            if not isinstance(value, dict) or list(value) != [sequence]:
                issues.append({"file": member, "issue": "JSON key differs from filename"})
                continue
            description = value[sequence]
            if not isinstance(description, str) or not description.strip():
                issues.append({"file": member, "issue": "Missing or non-string description"})
                continue
            if sequence in annotations:
                issues.append({"file": member, "issue": "Duplicate sequence ID"})
            annotations[sequence] = description.strip()

    joined = []
    for split in ("train", "test"):
        records = joblib.load(DATA / f"{split}_diffusion_manip_seq_joints24.p")
        for value in records.values():
            sequence = value["seq_name"]
            joined.append({
                "sequence": sequence,
                "split": split,
                "object": sequence.split("_")[1],
                "frames": len(value["trans"]),
                "has_text": int(sequence in annotations),
                "description": annotations.get(sequence, ""),
            })
    joined.sort(key=lambda row: (row["split"], row["sequence"]))
    write_csv(OUTPUT / "sequence_text.csv",
              ["sequence", "split", "object", "frames", "has_text", "description"], joined)

    grouped = defaultdict(lambda: [0, 0])
    for row in joined:
        for group in [(row["split"], "ALL"), (row["split"], row["object"])]:
            grouped[group][0] += 1
            grouped[group][1] += row["has_text"]
    coverage = [
        {"split": split, "object": object_name, "sequences": total,
         "with_text": matched, "coverage_pct": round(100 * matched / total, 2)}
        for (split, object_name), (total, matched) in sorted(grouped.items())
    ]
    write_csv(OUTPUT / "coverage_by_object.csv",
              ["split", "object", "sequences", "with_text", "coverage_pct"], coverage)

    released = {row["sequence"] for row in joined}
    extra = [{"sequence": name, "description": annotations[name]}
             for name in sorted(annotations.keys() - released)]
    write_csv(OUTPUT / "annotations_without_sequence.csv", ["sequence", "description"], extra)
    write_csv(OUTPUT / "annotation_issues.csv", ["file", "issue"], issues)
    print(f"released={len(joined)} annotated={sum(row['has_text'] for row in joined)} "
          f"missing={len(joined)-sum(row['has_text'] for row in joined)} "
          f"extra={len(extra)} issues={len(issues)}")
    for row in coverage:
        if row["object"] == "ALL":
            print(f"{row['split']}: {row['with_text']}/{row['sequences']} ({row['coverage_pct']}%)")
    print(f"output={OUTPUT}")


if __name__ == "__main__":
    main()
