"""Bind semantic holds and the training index to the actual reviewed records."""
import json

from counterfactual.critic_api import FEATURE_VERSION, sha256
from counterfactual.protocol_checks import canonical_hash


def load_review_status(path, reviews_path):
    status = json.loads(path.read_text(encoding='utf-8'))
    if status.get('source_reviews_sha256') != sha256(reviews_path):
        raise ValueError('Semantic status must bind the exact consolidated review file')
    holds = status.get('excluded_from_confirmed_negative_ids')
    if not isinstance(holds, list) or any(not isinstance(s, str) or not s.strip() for s in holds):
        raise ValueError('Explicit semantic hold list required (empty only after resolution)')
    if len(holds) != len(set(holds)) or status.get('held_sample_count') != len(holds):
        raise ValueError('Semantic hold count/list mismatch')
    return status


def validate_reviewed_index(entries, audit):
    expected = {e['sample_id']: {**e, 'label': e['label_error'], 'feature_version': FEATURE_VERSION}
                for e in audit['eligible_samples']}
    actual = {e['sample_id']: e for e in entries}
    if len(actual) != len(entries) or not entries or set(actual) != set(expected):
        raise ValueError('Training index membership differs from reviewed eligible samples')
    for sid, entry in actual.items():
        if canonical_hash(entry) != canonical_hash(expected[sid]):
            raise ValueError('Training entry differs from reviewed label/provenance: ' + sid)
