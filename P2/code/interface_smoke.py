"""Run from an exported package using an explicitly supplied trusted checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.output.exists() or args.output.resolve().is_relative_to(root):
        raise ValueError('Use a new output file outside the immutable package')
    manifest = json.loads((root / 'package_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('Package file mismatch: ' + name)
    descriptor = json.loads((root / 'checkpoint_descriptor.json').read_text(encoding='utf-8'))
    if digest(args.checkpoint) != descriptor['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from the known local baseline; refusing to load')
    from counterfactual import critic_api
    from models import interaction_critic
    from counterfactual.generator_adapter import load_features
    from counterfactual.benchmark import validate_manifest
    from counterfactual.protocol_checks import check_pair_protocol
    from counterfactual.paired_quality import evaluate_quality
    # Imports above must resolve to exported code, not the source workspace.
    for module in (critic_api, interaction_critic):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Import escaped package')
    torch.set_num_threads(2)
    scorer = critic_api.CriticScorer(args.checkpoint)
    arrays = []
    for frames, distance in ((12, .02), (37, .08), (96, .15), (121, .04)):
        x = np.zeros((frames, 13), dtype=np.float32)
        x[:, :2] = distance
        x[:, 2:4] = float(distance < .05)
        x[:, 6] = np.linspace(0, .01, frames)
        arrays.append(x)
    batch = scorer.score_many(arrays, feature_version=critic_api.FEATURE_VERSION, batch_size=2)
    single = [scorer.score_features(x, feature_version=critic_api.FEATURE_VERSION) for x in arrays]
    max_difference = max(abs(a['p_error'] - b['p_error']) for a, b in zip(batch, single))
    if max_difference > 1e-6:
        raise ValueError('Single/batch mismatch')
    for result in batch:
        if not (np.isfinite(result['p_error']) and 0 <= result['p_error'] <= 1
                and abs(result['p_error'] + result['p_valid'] - 1) < 1e-7
                and result['error_flag'] == (result['p_error'] >= result['threshold_error'])
                and result['calibrated_on_real_data'] is False):
            raise ValueError('Score semantics violated')
    linked = critic_api.probabilities(np.array([-2., 0., 2.]), .5)
    if not linked[0]['p_error'] < linked[1]['p_error'] < linked[2]['p_error']:
        raise ValueError('p_error direction violated')
    for features, version in [(np.zeros((1, 13)), critic_api.FEATURE_VERSION),
                              (np.full((5, 13), np.nan), critic_api.FEATURE_VERSION),
                              (arrays[0], 'unknown')]:
        try:
            scorer.score_features(features, feature_version=version)
        except ValueError:
            pass
        else:
            raise ValueError('Invalid input was accepted')
    try:
        check_pair_protocol([], json.loads((root / 'protocols/pair_protocol.template.json').read_text()))
    except ValueError:
        pass
    else:
        raise ValueError('Pending C protocol was accepted')
    result = {'status': 'passed', 'scope': 'exported-package CPU interface smoke; artificial unlabeled fixtures',
              'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
              'device': 'cpu', 'package_manifest_sha256': digest(root / 'package_manifest.json'),
              'checkpoint_sha256': scorer.checkpoint_sha256, 'fixtures': len(arrays),
              'single_batch_max_difference': max_difference, 'complement_and_threshold_checks': True,
              'invalid_input_checks': True, 'pending_C_protocol_rejected': True,
              'real_generator_transfer_test': False, 'training_performed': False,
              'human_budget_consumed': 0, 'checkpoint_was_external_to_bundle': not args.checkpoint.resolve().is_relative_to(root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
