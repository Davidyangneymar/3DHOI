"""Issue the user-approved 12-item targeted semantic review packet.

The original budget34 plan and frozen v1 are read-only inputs. No annotation is
generated here. A further six primary reviews are reserved, not selected or
issued, until the first packet has been assessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from counterfactual.audit_reviews import audit, required_secondaries
from counterfactual.prepare_review_budget import focus
from counterfactual.review_mechanics import mechanics_for


APPROVED_IDS = (
    "sub9_mop_039__clean__none",
    "sub15_largetable_015__premature_release__low",
    "sub13_mop_017__object_motion_without_contact__medium",
    "sub15_largetable_015__clean__none",
    "sub12_vacuum_055__premature_release__medium",
    "sub1_vacuum_008__object_motion_without_contact__low",
    "sub15_suitcase_052__spatial_contact_shift__low",
    "sub15_plasticbox_020__clean__none",
    "sub15_suitcase_054__premature_release__medium",
    "sub15_plasticbox_020__object_motion_without_contact__medium",
    "sub15_suitcase_016__spatial_contact_shift__medium",
    "sub8_vacuum_003__premature_release__high",
)
PACKET_ID = "targeted_val12_20261010"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def plan(root: Path) -> tuple[list[dict], dict]:
    original = root / "outputs/semantic_review_v11"
    old_plan_path = root / "outputs/semantic_review_budget34_20261003/plan_manifest.json"
    index_path = root / "counterfactual/v1/index.jsonl"
    reviews_path = root / "reviews/incoming/budget34_batch02_secondary_20261003/combined_reviews_50.jsonl"
    status_path = reviews_path.with_name("review_status.json")
    entries = json.loads((original / "queue.json").read_text(encoding="utf-8"))
    index = load_jsonl(index_path)
    old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
    records = load_jsonl(reviews_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (status.get("source_reviews_sha256") != digest(reviews_path)
            or status.get("held_sample_count") != len(status.get("excluded_from_confirmed_negative_ids", []))):
        raise ValueError("Review status is not bound to the combined records")
    audit(entries, records, review_status=status)

    if len(entries) != len(index) or {e["sample_id"] for e in entries} != {e["sample_id"] for e in index}:
        raise ValueError("Full queue does not match the frozen index")
    index_by_id = {e["sample_id"]: e for e in index}
    by_id = {e["sample_id"]: e for e in entries}
    if any(any(e.get(k) != index_by_id[e["sample_id"]].get(k) for k in
               ("sha256", "split", "source_sequence", "type", "severity", "frames")) for e in entries):
        raise ValueError("Review queue differs from the frozen index")
    if len(APPROVED_IDS) != 12 or len(set(APPROVED_IDS)) != 12 or not set(APPROVED_IDS) <= set(by_id):
        raise ValueError("Approved sample list missing or duplicated")
    primaries = {r["sample_id"] for r in records if r["round"] == "primary"}
    if len(primaries) != 36 or set(APPROVED_IDS) & primaries:
        raise ValueError("Primary budget or unreviewed status changed; replan explicitly")
    batch03 = set(old_plan["batches"][2]["sample_ids"])
    if len(batch03) != 12 or len(set(APPROVED_IDS) & batch03) != 6:
        raise ValueError("Proposal no longer replaces exactly six batch03 samples")
    if sum(e["split"] == "val" for e in (by_id[sid] for sid in APPROVED_IDS)) != 7:
        raise ValueError("Targeted validation split changed")
    if sum(e["split"] == "train" for e in (by_id[sid] for sid in APPROVED_IDS)) != 5:
        raise ValueError("Targeted training split changed")

    required = required_secondaries(entries)
    selected = []
    for sid in APPROVED_IDS:
        entry = {**by_id[sid]}
        entry["second_review_required"] = (sid in required or entry.get("second_review_required", False)
                                           or entry["split"] == "val" or entry["type"] == "spatial_contact_shift")
        entry["review_focus"] = focus(entry)
        entry["review_mechanics"] = mechanics_for(entry, root / "counterfactual/v1/samples" / f"{sid}.npz")
        selected.append(entry)

    evidence = {
        "original_plan_sha256": digest(old_plan_path),
        "full_queue_sha256": digest(original / "queue.json"),
        "frozen_index_sha256": digest(index_path),
        "input_reviews_sha256": digest(reviews_path),
        "review_status_sha256": digest(status_path),
        "prior_primary_count": len(primaries),
        "retained_original_batch03_ids": sorted(set(APPROVED_IDS) & batch03),
        "displaced_original_batch03_ids": sorted(batch03 - set(APPROVED_IDS)),
        "new_ids": sorted(set(APPROVED_IDS) - batch03),
    }
    return selected, evidence


def build(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Packet already exists; refusing to overwrite it")
    selected, evidence = plan(root)
    original = root / "outputs/semantic_review_v11"
    payload_text = (original / "queue.js").read_text(encoding="utf-8")
    payload = json.loads(payload_text.removeprefix("window.queue=").removesuffix(";"))
    if "entries" not in payload or len(payload["entries"]) != 436:
        raise ValueError("Unexpected full review payload")
    context = {
        "packet_id": PACKET_ID,
        "storage_key": "hoi_" + PACKET_ID,
        "title": "定向语义首审 · 12 条",
        "required_reviewer": "ymh",
        "required_round": "primary",
        "semantic_only": True,
        "instructions": "本批为已确认的12条定向首审，原batch03不再作为当前入口。请先看完整回放，再结合动作和阶段判断；定位片段和距离只是线索。正常、错误、不确定、样本无效都可以选，不为了补齐类别而勉强判错。空间接触偏移请说明为什么该位置与动作不相容；保存后导出JSONL。模型/规则分数不显示。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".targeted_review_", dir=output.parent) as temp_name:
        stage = Path(temp_name) / "packet"
        (stage / "samples").mkdir(parents=True)
        (stage / "queue.js").write_text(
            "window.queue=" + json.dumps({**payload, "entries": selected, "review_context": context}, ensure_ascii=True) + ";",
            encoding="utf-8",
        )
        (stage / "queue.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        assets = {e["sample_id"] for e in selected} | {e["source_sequence"] + "__clean__none" for e in selected}
        for sid in sorted(assets):
            source = original / "samples" / f"{sid}.js"
            target = stage / "samples" / source.name
            shutil.copy2(source, target)
            if digest(source) != digest(target):
                raise ValueError("Review sample asset copy mismatch: " + sid)
        shutil.copy2(original / "plotly.min.js", stage / "plotly.min.js")
        for name, target in (("review_template.html", "index.html"),
                             ("review_app.js", "review_app.js"),
                             ("review_form.js", "review_form.js")):
            shutil.copy2(root / "counterfactual" / name, stage / target)
        manifest = {
            "packet_id": PACKET_ID,
            "status": "issued_primary_packet",
            "approved_date_local": "2026-10-10",
            "approved_primary_ids": list(APPROVED_IDS),
            "count": len(selected),
            "split_counts": dict(Counter(e["split"] for e in selected)),
            "type_severity_counts": dict(Counter(e["type"] + "/" + e["severity"] for e in selected)),
            "unique_source_count": len({e["source_sequence"] for e in selected}),
            "required_secondary_ids": [e["sample_id"] for e in selected if e["second_review_required"]],
            "secondary_policy": "all validation candidates and all spatial shifts plus parent queue requirements; verify independently before any model selection",
            "previous_additional_primary_cap": 34,
            "previous_additional_primary_submitted": 22,
            "this_packet_additional_primary": 12,
            "optional_supplemental_primary_cap": 6,
            "optional_supplemental_issued": False,
            "automatic_refill": False,
            "full_v1_approved": False,
            "training_ready": False,
            "selection_method": "User-approved replacement of six unreviewed batch03 samples with four validation low/medium corruption candidates and two paired validation clean samples; nonrandom diagnostic subset",
            "source_evidence": evidence,
            "sample_npz_sha256": {e["sample_id"]: e["sha256"] for e in selected},
            "asset_sha256": {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()},
            "browser_visual_verification": False,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if output.exists():
            raise ValueError("Packet destination appeared while building")
        stage.rename(output)
    return manifest


def verify(root: Path, output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    entries = json.loads((output / "queue.json").read_text(encoding="utf-8"))
    payload_text = (output / "queue.js").read_text(encoding="utf-8")
    payload = json.loads(payload_text.removeprefix("window.queue=").removesuffix(";"))
    if ([e["sample_id"] for e in entries] != list(APPROVED_IDS)
            or payload["entries"] != entries
            or payload["review_context"]["packet_id"] != PACKET_ID
            or manifest["approved_primary_ids"] != list(APPROVED_IDS)):
        raise ValueError("Packet queue or context differs from approval")
    if any(set(e) & {"critic_score", "rule_score", "p_error", "p_valid"} for e in entries):
        raise ValueError("Model scores leaked into primary queue")
    expected_assets = manifest["asset_sha256"]
    actual_assets = {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*")
                     if p.is_file() and p.name != "manifest.json"}
    if actual_assets != expected_assets:
        raise ValueError("Packet assets differ from issued manifest")
    originals = {
        "original_plan_sha256": root / "outputs/semantic_review_budget34_20261003/plan_manifest.json",
        "full_queue_sha256": root / "outputs/semantic_review_v11/queue.json",
        "frozen_index_sha256": root / "counterfactual/v1/index.jsonl",
        "input_reviews_sha256": root / "reviews/incoming/budget34_batch02_secondary_20261003/combined_reviews_50.jsonl",
        "review_status_sha256": root / "reviews/incoming/budget34_batch02_secondary_20261003/review_status.json",
    }
    if any(digest(path) != manifest["source_evidence"][key] for key, path in originals.items()):
        raise ValueError("An input changed since packet issue")
    if any(digest(root / "counterfactual/v1/samples" / f"{e['sample_id']}.npz") != e["sha256"]
           for e in entries):
        raise ValueError("Frozen NPZ differs from packet queue")
    return {"verified": True, "count": len(entries), "asset_count": len(actual_assets),
            "split_counts": dict(Counter(e["split"] for e in entries))}


def build_secondary(root: Path, primary_output: Path, secondary_output: Path) -> dict:
    if secondary_output.exists():
        raise ValueError("Secondary packet already exists; refusing to overwrite it")
    verify(root, primary_output)
    primary_manifest = json.loads((primary_output / "manifest.json").read_text(encoding="utf-8"))
    payload_text = (primary_output / "queue.js").read_text(encoding="utf-8")
    payload = json.loads(payload_text.removeprefix("window.queue=").removesuffix(";"))
    required = set(primary_manifest["required_secondary_ids"])
    selected = [e for e in payload["entries"] if e["sample_id"] in required]
    if len(selected) != len(required) or len(selected) != 9:
        raise ValueError("Required secondary set changed")
    packet_id = PACKET_ID + "_secondary"
    context = {
        "packet_id": packet_id,
        "storage_key": "hoi_" + packet_id,
        "title": "定向语义独立二审 · 9 条",
        "required_round": "secondary",
        "excluded_reviewers": ["ymh"],
        "semantic_only": True,
        "instructions": "请由不同于首审者的审核人独立判断9条样本。此页不含首审结论、备注或模型分数。先看完整回放，再选动作、阶段、结论及理由；信息不足可选不确定。完成后导出本人的JSONL，勿参考首审票据。",
    }
    secondary_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".targeted_secondary_", dir=secondary_output.parent) as temp_name:
        stage = Path(temp_name) / "packet"
        (stage / "samples").mkdir(parents=True)
        (stage / "queue.js").write_text(
            "window.queue=" + json.dumps({**payload, "entries": selected, "review_context": context}, ensure_ascii=True) + ";",
            encoding="utf-8",
        )
        (stage / "queue.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        assets = {e["sample_id"] for e in selected} | {e["source_sequence"] + "__clean__none" for e in selected}
        for sid in sorted(assets):
            source = primary_output / "samples" / f"{sid}.js"
            target = stage / "samples" / source.name
            shutil.copy2(source, target)
            if digest(source) != digest(target):
                raise ValueError("Secondary asset copy mismatch: " + sid)
        for name in ("index.html", "review_app.js", "review_form.js", "plotly.min.js"):
            shutil.copy2(primary_output / name, stage / name)
        manifest = {
            "packet_id": packet_id,
            "status": "issued_independent_secondary_packet",
            "count": len(selected),
            "sample_ids": [e["sample_id"] for e in selected],
            "excluded_primary_reviewer": "ymh",
            "primary_manifest_sha256": digest(primary_output / "manifest.json"),
            "primary_queue_sha256": digest(primary_output / "queue.json"),
            "no_primary_votes_embedded": True,
            "asset_sha256": {p.relative_to(stage).as_posix(): digest(p) for p in stage.rglob("*") if p.is_file()},
            "browser_visual_verification": False,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if secondary_output.exists():
            raise ValueError("Secondary destination appeared while building")
        stage.rename(secondary_output)
    return manifest


def verify_secondary(root: Path, primary_output: Path, secondary_output: Path) -> dict:
    verify(root, primary_output)
    primary_manifest = json.loads((primary_output / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((secondary_output / "manifest.json").read_text(encoding="utf-8"))
    entries = json.loads((secondary_output / "queue.json").read_text(encoding="utf-8"))
    payload_text = (secondary_output / "queue.js").read_text(encoding="utf-8")
    payload = json.loads(payload_text.removeprefix("window.queue=").removesuffix(";"))
    expected = primary_manifest["required_secondary_ids"]
    if ([e["sample_id"] for e in entries] != expected
            or payload["entries"] != entries
            or manifest["sample_ids"] != expected
            or payload["review_context"].get("required_round") != "secondary"
            or payload["review_context"].get("excluded_reviewers") != ["ymh"]
            or manifest["primary_manifest_sha256"] != digest(primary_output / "manifest.json")
            or manifest["primary_queue_sha256"] != digest(primary_output / "queue.json")):
        raise ValueError("Independent secondary queue or provenance differs")
    if any(set(e) & {"critic_score", "rule_score", "p_error", "p_valid", "primary_decision"} for e in entries):
        raise ValueError("Answer or model score leaked to independent secondary")
    actual_assets = {p.relative_to(secondary_output).as_posix(): digest(p)
                     for p in secondary_output.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual_assets != manifest["asset_sha256"]:
        raise ValueError("Independent secondary assets differ from manifest")
    return {"verified": True, "count": len(entries), "asset_count": len(actual_assets),
            "primary_votes_embedded": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic_review_targeted_20261010"))
    parser.add_argument("--verify", action="store_true", help="Check an already issued packet without changing it")
    parser.add_argument("--issue-secondary", action="store_true", help="Issue score/answer-free independent secondary packet")
    parser.add_argument("--verify-secondary", action="store_true", help="Check an already issued independent secondary packet")
    parser.add_argument("--secondary-output", type=Path,
                        default=Path("outputs/semantic_review_targeted_20261010_secondary"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    if args.verify_secondary:
        secondary_output = args.secondary_output if args.secondary_output.is_absolute() else root / args.secondary_output
        print(json.dumps(verify_secondary(root, output, secondary_output), ensure_ascii=False, indent=2))
        return
    if args.issue_secondary:
        secondary_output = args.secondary_output if args.secondary_output.is_absolute() else root / args.secondary_output
        result = build_secondary(root, output, secondary_output)
        print(json.dumps({k: result[k] for k in ("packet_id", "count", "sample_ids")}, ensure_ascii=False, indent=2))
        return
    if args.verify:
        print(json.dumps(verify(root, output), ensure_ascii=False, indent=2))
        return
    result = build(root, output)
    print(json.dumps({k: result[k] for k in ("packet_id", "count", "split_counts", "required_secondary_ids", "optional_supplemental_issued")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
