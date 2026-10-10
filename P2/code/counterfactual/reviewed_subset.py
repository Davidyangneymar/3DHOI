"""Explicit partial-review release policy; full-queue audit remains authoritative."""
from collections import Counter

from counterfactual.protocol_checks import canonical_hash

POLICY_VERSION = 'reviewed_subset_v1'


def validate_parent_queue(queue, parent_entries):
    originals = {e['sample_id']: e for e in parent_entries}
    if len(queue) != len(parent_entries) or len(originals) != len(parent_entries):
        raise ValueError('Review queue must cover the complete frozen v1 index')
    if {e['sample_id'] for e in queue} != set(originals):
        raise ValueError('Review queue membership differs from frozen parent')
    for entry in queue:
        original = originals[entry['sample_id']]
        if canonical_hash({k: entry.get(k) for k in original}) != canonical_hash(original):
            raise ValueError('Review queue differs from frozen parent label/split/hash')


def training_eligibility(entries):
    """Necessary engineering gates, not a claim of statistical adequacy."""
    coverage, blockers = {}, []
    sources = {}
    for e in entries:
        sources.setdefault(e['source_sequence'], set()).add(e['split'])
    if any(len(v) != 1 for v in sources.values()):
        blockers.append('source_leakage')
    for split in ('train', 'val', 'test'):
        all_rows = [e for e in entries if e['split'] == split]
        usable = [e for e in all_rows if split == 'test' or e['severity'] != 'high']
        coverage[split] = {
            'all_count': len(all_rows), 'usable_count': len(usable),
            'all_class_counts': dict(Counter(str(e['label_error']) for e in all_rows)),
            'usable_class_counts': dict(Counter(str(e['label_error']) for e in usable)),
            'independent_sources': len({e['source_sequence'] for e in all_rows}),
            'usable_sources_by_class': {str(y): len({e['source_sequence'] for e in usable if e['label_error'] == y}) for y in (0, 1)},
            'types': dict(Counter(e['type'] for e in all_rows)),
            'severities': dict(Counter(e['severity'] for e in all_rows))}
        if {e['label_error'] for e in usable} != {0, 1}:
            blockers.append(f'{split}_needs_both_classes' + ('_after_withholding_high' if split != 'test' else ''))
    return {'training_ready': not blockers, 'blockers': blockers, 'coverage': coverage,
            'withheld_severity': 'high', 'split_policy': 'unchanged_v1_source_splits',
            'statistical_adequacy_verified': False, 'test_status': 'previously_exposed_regression_only'}


def select_reviewed_subset(queue, full_audit):
    """Do not rerun audit on a reduced queue: that changes secondary obligations."""
    included = []
    exclusions = []
    eligible = {e['sample_id']: e for e in full_audit['eligible_samples']}
    for entry in queue:
        sid = entry['sample_id']
        status = full_audit['statuses'][sid]
        if status not in ('valid', 'error'):
            reason = status
        elif entry['type'] == 'spatial_contact_shift':
            # Existing vote fields do not certify an action-incompatible region.
            # Supporting evidence needs a separately reviewed policy revision.
            reason = 'action_region_evidence_not_validated'
        else:
            included.append(eligible[sid])
            continue
        exclusions.append({'sample_id': sid, 'review_status': status, 'reason': reason,
                           'sample_sha256': entry['sha256'], 'split': entry['split']})
    return {'policy_version': POLICY_VERSION, 'review_scope': 'reviewed-subset',
            'ready_for_subset_freeze': bool(included),
            'full_queue_ready': full_audit['ready_for_curated_freeze'],
            'sample_ids': [e['sample_id'] for e in included],
            'eligible_samples': included, 'exclusions': exclusions,
            'diagnostic_only_ids': [e['sample_id'] for e in exclusions if e['reason'] == 'action_region_evidence_not_validated'],
            'training_eligibility': training_eligibility(included),
            'selection_basis': 'all eligible non-spatial votes from full audit; no score-based selection',
            'population_quality_estimate': False}
