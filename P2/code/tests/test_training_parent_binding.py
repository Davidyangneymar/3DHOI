"""Training parent-binding fixtures; every successful gate stops before fitting."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from counterfactual.audit_reviews import audit
from counterfactual.critic_api import FEATURE_VERSION, sha256
from counterfactual.reviewed_subset import select_reviewed_subset
from counterfactual import train_reviewed


class BeforeModelConstruction(RuntimeError):
    """A sentinel, never a trained model."""


def write_fixture(root, scope='full', mutation=None):
    parent = root/'counterfactual/v1'; (parent/'samples').mkdir(parents=True)
    dataset = root/'reviewed'; (dataset/'samples').mkdir(parents=True)
    queue = []
    for split in ('train', 'val', 'test'):
        for label in (0, 1):
            sid = f'{split}_{label}'
            sample = parent/'samples'/(sid+'.npz')
            np.savez(sample, features=np.zeros((3, 13), np.float32))
            entry = dict(sample_id=sid, source_sequence=sid, split=split, label=label, frames=3,
                         severity='none' if label == 0 else 'low', type='clean' if label == 0 else 'premature_release',
                         sha256=sha256(sample))
            queue.append(entry)
            (dataset/'samples'/(sid+'.npz')).write_bytes(sample.read_bytes())
    original = copy.deepcopy(queue)
    if mutation == 'reduced_queue':
        original.append({**original[0], 'sample_id': 'unreviewed_parent', 'source_sequence': 'unreviewed_parent'})
    (parent/'index.jsonl').write_text('\n'.join(json.dumps(e) for e in original), encoding='utf-8')
    (parent/'manifest.json').write_text(json.dumps({'index_sha256': sha256(parent/'index.jsonl')}), encoding='utf-8')
    if mutation == 'split': queue[0]['split'] = 'test'
    if mutation == 'label': queue[0]['label'] = 1
    if mutation == 'sample_hash': queue[0]['sha256'] = '0'*64
    votes = []
    for e in queue:
        for reviewer, round_name in [('reviewer-one', 'primary'), ('reviewer-two', 'secondary')]:
            votes.append(dict(sample_id=e['sample_id'], sample_sha256=e['sha256'], reviewer=reviewer,
                              round=round_name, decision='valid' if e['type'] == 'clean' else 'error',
                              action='carry', phase='transport', start_frame=0, end_frame=2,
                              reason='Synthetic test fixture only'))
    (dataset/'review_records.jsonl').write_text('\n'.join(json.dumps(v) for v in votes), encoding='utf-8')
    status = dict(source_reviews_sha256=sha256(dataset/'review_records.jsonl'),
                  excluded_from_confirmed_negative_ids=[], held_sample_count=0)
    result = audit(queue, votes, review_status=status)
    selection = select_reviewed_subset(queue, result) if scope == 'reviewed-subset' else None
    entries = [{**e, 'label': e['label_error'], 'feature_version': FEATURE_VERSION}
               for e in (selection or result)['eligible_samples']]
    (dataset/'index.jsonl').write_text('\n'.join(json.dumps(e) for e in entries), encoding='utf-8')
    (dataset/'adjudications.jsonl').write_text('', encoding='utf-8')
    for filename, value in [('queue.json', queue), ('audit.json', result), ('review_status.json', status)]:
        (dataset/filename).write_text(json.dumps(value), encoding='utf-8')
    manifest = dict(feature_version=FEATURE_VERSION, review_scope=scope,
                    parent_manifest_sha256=sha256(parent/'manifest.json'))
    for filename, key in [('queue.json', 'queue_sha256'), ('audit.json', 'audit_sha256'),
                          ('review_status.json', 'review_status_sha256'), ('review_records.jsonl', 'review_records_sha256'),
                          ('adjudications.jsonl', 'adjudications_sha256'), ('index.jsonl', 'index_sha256')]:
        manifest[key] = sha256(dataset/filename)
    if selection:
        (dataset/'subset.json').write_text(json.dumps(selection), encoding='utf-8')
        manifest['subset_sha256'] = sha256(dataset/'subset.json')
    if mutation == 'missing_parent_hash': manifest.pop('parent_manifest_sha256')
    if mutation == 'parent_manifest': manifest['parent_manifest_sha256'] = '0'*64
    if mutation == 'parent_index':
        with (parent/'index.jsonl').open('a', encoding='utf-8') as stream: stream.write('\n')
    (dataset/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return dataset


class TrainingParentBindingTests(unittest.TestCase):
    def run_entry(self, root, dataset, module=train_reviewed):
        output = root/'training'
        with patch('models.interaction_critic.ROOT', root), \
             patch.object(sys, 'argv', ['train', str(dataset), '--output', str(output)]), \
             patch.object(module, 'InteractionCritic', side_effect=BeforeModelConstruction('STOP BEFORE MODEL')) as model:
            try:
                module.main()
            finally:
                self.model_called = model.called
        return output

    def test_both_modes_reject_small_queue_and_rehashed_parent_fields(self):
        for scope in ('full', 'reviewed-subset'):
            for mutation in ('reduced_queue', 'split', 'label', 'sample_hash'):
                with self.subTest(scope=scope, mutation=mutation), tempfile.TemporaryDirectory() as td:
                    root = Path(td); dataset = write_fixture(root, scope, mutation)
                    with self.assertRaisesRegex(ValueError, 'complete frozen|parent label/split/hash'):
                        self.run_entry(root, dataset)
                    self.assertFalse(self.model_called)
                    self.assertFalse((root/'training').exists())

    def test_both_modes_require_bound_parent_manifest_and_index(self):
        for scope in ('full', 'reviewed-subset'):
            for mutation in ('missing_parent_hash', 'parent_manifest', 'parent_index'):
                with self.subTest(scope=scope, mutation=mutation), tempfile.TemporaryDirectory() as td:
                    root = Path(td); dataset = write_fixture(root, scope, mutation)
                    with self.assertRaises(SystemExit): self.run_entry(root, dataset)
                    self.assertFalse(self.model_called)
                    self.assertFalse((root/'training').exists())

    def test_valid_fixtures_pass_gates_but_stop_before_model_construction(self):
        for scope in ('full', 'reviewed-subset'):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as td:
                root = Path(td); dataset = write_fixture(root, scope)
                with self.assertRaises(BeforeModelConstruction): self.run_entry(root, dataset)
                self.assertTrue(self.model_called)
                self.assertFalse(any((root/'training').rglob('*.pt')))


if __name__ == '__main__': unittest.main()
