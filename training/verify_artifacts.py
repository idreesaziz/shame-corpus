"""Validate saved artifacts and provenance without loading models or generating text."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2]/'outputs/modal_training_20261111')
    parser.add_argument('--require-all', action='store_true')
    parser.add_argument('--recipe', type=Path, default=Path(__file__).parent/'recipe.json')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    errors, checked, pending = [], [], []
    root = args.root.resolve()
    for key, base in recipe['models'].items():
        for condition in recipe['conditions']:
            for seed in recipe['seeds']:
                name = f'{key}/{condition}/seed-{seed}'
                path = root/'artifacts'/name
                manifest = path/'manifest.json'
                if not manifest.exists() or not (path/'adapter/adapter_model.safetensors').exists():
                    pending.append(name)
                    continue
                try:
                    row = json.loads(manifest.read_text())
                    assert row['status'] == 'complete'
                    assert row['model_key'] == key and row['condition'] == condition and row['seed'] == seed
                    assert row['base_model_id'] == base['model_id'] and row['base_revision'] == base['revision']
                    assert row['corpus_sha256'] == recipe['corpus_sha256']
                    assert row['config'] == recipe
                    assert row['examples_seen'] == 2000 and row['epochs_completed'] == 1
                    assert row['optimizer_steps'] == 250
                    assert row['adapter_absolute_update'] > 0
                    assert row['adapter_reopened_successfully'] and row['frozen_base_sample_unchanged']
                    for filename, expected in row['artifacts'].items():
                        actual = hashlib.sha256((path/'adapter'/filename).read_bytes()).hexdigest()
                        assert actual == expected, 'Hash mismatch: '+filename
                    config = json.loads((path/'adapter/adapter_config.json').read_text())
                    assert config['base_model_name_or_path'] == base['model_id']
                    assert config['revision'] == base['revision']
                    assert config['r'] == recipe['lora_rank'] and config['lora_alpha'] == recipe['lora_alpha']
                    weights = path/'adapter/adapter_model.safetensors'
                    with weights.open('rb') as file:
                        length = struct.unpack('<Q', file.read(8))[0]
                        tensors = json.loads(file.read(length))
                    tensors = {k:v for k,v in tensors.items() if k != '__metadata__'}
                    assert tensors and all('lora_' in k for k in tensors)
                    assert all(v['data_offsets'][1] <= weights.stat().st_size-8-length for v in tensors.values())
                    if key in ('gemma','qwen9','mistral8'): assert all('language_model' in k for k in tensors)
                    steps = [json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
                    assert steps[-1]['step'] == 250 and steps[-1]['examples_seen'] == 2000
                    checked.append({'run':name, 'adapter_sha256':row['artifacts']['adapter_model.safetensors'],
                                    'tensor_count':len(tensors), 'seconds':row['cumulative_seconds']})
                except Exception as exc:
                    errors.append({'run':name, 'error':str(exc) or type(exc).__name__})
    expected=len(recipe['models'])*len(recipe['conditions'])*len(recipe['seeds'])
    report = {'verified_runs':len(checked),'expected_runs':expected,'pending_runs':pending,'errors':errors,'runs':checked,
              'scope':'Artifact integrity and training provenance only; no behavioral evaluation.'}
    (root/'artifact_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if errors or (args.require_all and pending): raise SystemExit(1)


if __name__ == '__main__': main()
