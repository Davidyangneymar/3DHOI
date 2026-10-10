"""Fail-closed checks for externally supplied condition splits and candidate pools."""
from __future__ import annotations

import hashlib
import json


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def valid_error_label(value):
    return value is None or type(value) is int and value in (0, 1) or type(value) is str and value == 'uncertain'


def check_pair_protocol(rows, protocol):
    """A C-authored manifest states exposure; code checks it, not its truthfulness."""
    if protocol.get("status") != "frozen":
        raise ValueError("C pair protocol must be frozen before evaluation")
    for field in ("protocol_id", "owner", "generator_training_manifest_sha256", "critic_training_manifest_sha256"):
        require_text(protocol.get(field), field)
    train = protocol.get("training_pairs")
    unseen = protocol.get("unseen_pairs")
    if not isinstance(train, list) or not isinstance(unseen, list) or not unseen:
        raise ValueError("training_pairs and nonempty unseen_pairs lists are required")
    if any(not isinstance(p, list) or len(p) != 2 or any(not isinstance(v, str) or not v.strip() for v in p) for p in train + unseen):
        raise ValueError("Pairs must be [action,object] string arrays")
    train, unseen = set(map(tuple, train)), set(map(tuple, unseen))
    if train & unseen:
        raise ValueError("Unseen pairs appear in declared training pairs")
    assignments = protocol.get("conditions", {})
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Frozen condition assignments required")
    if any(c.get('pair_status') == 'unseen' for c in assignments.values()) and protocol.get('pretraining_exposure_verified') is not True:
        raise ValueError('Unseen claims require verified training/pretraining exposure')
    for row in rows:
        condition = assignments.get(row['condition_id'])
        if condition is None:
            raise ValueError("Condition absent from C protocol")
        pair = (row.get('action'), row.get('object'))
        if (condition.get('action'), condition.get('object')) != pair or condition.get('split') != row['split']:
            raise ValueError("Condition action/object/split disagrees with C protocol")
        claim = condition.get('pair_status')
        if claim == 'unseen':
            if pair not in unseen or row['split'] != 'test':
                raise ValueError("Unseen pairs must be explicitly declared and reserved for test")
        elif claim == 'seen':
            if pair not in train:
                raise ValueError("Seen pair is absent from declared training pairs")
        else:
            raise ValueError("pair_status must be seen/unseen")
    return {'protocol_sha256': canonical_hash(protocol), 'checked_conditions': len({r['condition_id'] for r in rows}),
            'declared_split_consistent': True,
            'pretraining_exposure_verified': protocol.get('pretraining_exposure_verified') is True}
