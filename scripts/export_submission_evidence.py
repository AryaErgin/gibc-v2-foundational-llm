"""Read frozen local records; export a compact submission audit, never run models/tasks.

Writes only the explicitly named NEW report. No dataset libraries, evaluation
functions, network access, training or model forward pass are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = 'exp020-final-7p2b-train-20260905'
DATA = 'exp020-final-7p2b-data-rebuild-2'
TASKS = ('hellaswag', 'arc_easy', 'piqa', 'winogrande', 'wikitext103')
EXPECTED = {
    'checkpoint': '95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89',
    'config': '25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace',
    'tokenizer': 'c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14',
    'summary': '31c795afce30b7060b58bd8aa5c8ff14aad98ccb79302c450c8f75d48a995d8a',
    'manifest': 'aedbbb8dcfe47c5b0d7de2ee052fbeea93232db5a917d09e568c394fe06eacc7',
    'stream': '94e39c09e7696a9668568802c37e6458799b47284efa566f74fb6792f571440e',
    'index': '4b47a02d0bfa793809b02adcc251eb2f3560217e1ddcc0c595a78906386e7a1f',
}


def digest(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while limit is None or limit > 0:
            block = f.read(8 * 1024 * 1024 if limit is None else min(limit, 8 * 1024 * 1024))
            if not block:
                if limit:
                    raise ValueError(f'Short prefix: {path}')
                break
            h.update(block)
            if limit is not None:
                limit -= len(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--logs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-stream', action='store_true', help='Read/hash the frozen 14.4GB stream and both historical prefixes')
    parser.add_argument('--checkpoint-tensors', action='store_true', help='Read tensors on CPU; count tied storage once, without constructing/executing a model')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite submission evidence: {args.output}')
    train, data = args.artifact_root / FINAL, args.artifact_root / DATA
    official = args.artifact_root / 'exp020-official-eval'
    paths = {
        'checkpoint': train/'checkpoints/checkpoint-step-219726.pt',
        'config': ROOT/'configs/exp020-final-7p2b-cosine.yaml',
        'tokenizer': data/'tokenizer/tokenizer.json', 'summary': train/'summary.json',
        'manifest': data/'manifest.json',
        'index': args.artifact_root/'exp012-full-data/cache/benchmarks/benchmark-ngrams.sqlite',
        'metrics': train/'metrics.jsonl', 'progress': train/'progress.jsonl',
        'runtime': args.logs_root/'exp020-evaluation-runtime-provenance-20260907.json',
        'preflight': args.logs_root/'exp020-official-eval-preflight-20260907.json',
        'suite': args.logs_root/'exp020-official-eval-suite.status.json',
        'sequence': official/'status/sequence.status.json',
        'preregistration': ROOT/'provenance/exp020-final-preregistration.json',
    }
    sources = {}
    for name, path in paths.items():
        actual = digest(path)
        if name in EXPECTED and actual != EXPECTED[name]:
            raise ValueError(f'Frozen hash mismatch: {name}: {actual}')
        sources[name] = {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}
        print('Verified', name, actual, flush=True)
    assert sources['checkpoint']['bytes'] == 598447091
    summary, manifest, suite, sequence = [load(paths[k]) for k in ('summary','manifest','suite','sequence')]
    assert summary['status'] == 'FULL HORIZON COMPLETE'
    assert summary['final_step'] == 219726 and summary['prediction_tokens'] == 7199981568
    assert summary['parameter_count'] == 49860480
    assert summary['next_sequence_index'] == 219726*64
    assert suite['state'] == sequence['state'] == 'succeeded' and suite['returncode'] == 0
    assert summary['git_commit'] == 'd88800733846c7a30e2044fa0a20f9e4a448f328'
    assert summary['data_manifest_sha256'] == sources['manifest']['sha256']
    stream = data/'train-token-stream.uint16'
    assert stream.stat().st_size == 14399963138 == (7199981568+1)*2
    if args.full_stream:
        assert digest(stream) == EXPECTED['stream']
        sources['stream'] = {'path':str(stream),'bytes':stream.stat().st_size,'sha256':EXPECTED['stream']}
        for name in ('exp011_prefix','exp012_prefix'):
            prefix = manifest[name]
            assert digest(stream, prefix['byte_count']) == prefix['expected_sha256'] == prefix['observed_sha256']
        print('Full stream and historical prefixes verified',flush=True)
    validation = []
    for g,e in zip(summary['validation_records'],summary['edu_validation_records'],strict=True):
        assert g['step'] == e['step']
        validation.append({'step':int(g['step']), 'tokens':int(g['step'])*32768,
                           'general':g['loss'],'edu':e['loss'],'combined':(g['loss']+e['loss'])/2})
    late = [r for r in validation if r['step']>=146484]
    assert min(late,key=lambda r:r['combined'])['step'] == 219726
    bins, losses, last, events = [], [], 0, {}
    with paths['metrics'].open() as f:
        for line in f:
            r= json.loads(line)
            if r.get('event') == 'run_start':
                assert r['start_step']==0 and r['requested_steps']==r['schedule_horizon_steps']==219726
                events['run_start']=r
            if r.get('event') != 'train':
                continue
            step=int(r['step'])
            assert step==last+1 and r['next_sequence_index']==step*64 and r['cumulative_tokens']==step*32768
            assert all(math.isfinite(r[k]) for k in ('loss','gradient_norm','learning_rate','tokens_per_second','paced_tokens_per_second'))
            losses.append(r['loss']);last=step
            if step % 1000 == 0 or step==219726:
                bins.append({'start_step':step-len(losses)+1,'end_step':step,'tokens':step*32768,'mean_train_loss':sum(losses)/len(losses)})
                losses=[]
    assert last==219726
    events['train_rows_verified']=last
    progress=[]
    with paths['progress'].open() as f:
        for line in f:
            r=json.loads(line)
            progress.append(r)
    events['first_durable_progress']=progress[0]
    events['last_durable_progress']=progress[-1]
    results={}
    for task in TASKS:
        path=official/('wikitext103.json' if task=='wikitext103' else f'lm_eval/{task}.json')
        raw=load(path); meta=raw['metadata']; item=next(x for x in sequence['tasks'] if x['task']==task)
        actual=digest(path); assert actual==item['sha256']
        assert meta['checkpoint_sha256']==EXPECTED['checkpoint'] and meta['config_sha256']==EXPECTED['config']
        assert meta['tokenizer_sha256']==EXPECTED['tokenizer'] and meta['selected_step']==219726
        assert meta['device']=='cpu' and meta['precision']=='fp32' and meta['cuda_available'] is False
        assert meta['batch_size']==16 and meta['num_fewshot']==0 and meta['context_length']==512
        sources[task]={'path':str(path),'sha256':actual,'bytes':path.stat().st_size}
        results[task]={'metadata':meta,'metrics':raw['result'] if task=='wikitext103' else raw['raw_lm_eval_result']['results'][task]}
        if task!='wikitext103':
            results[task]['sample_counts']=raw['raw_lm_eval_result']['n-samples'][task]
            results[task]['task_definition']=raw['raw_lm_eval_result']['configs'][task]
    for domain in ('general','edu'):
        p=data/f'{domain}_validation.pt'; previous=args.artifact_root/'exp012-full-data'/p.name
        assert digest(p)==digest(previous)
        sources[domain+'_validation']={'path':str(p),'sha256':digest(p),'bytes':p.stat().st_size,'byte_identical_to_exp012':True}
    tensor_count=None
    if args.checkpoint_tensors:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']=''
        import torch
        payload=torch.load(paths['checkpoint'],map_location='cpu',weights_only=False)
        assert payload['run_state']=={'step':219726,'tokens':7199981568,'next_sequence_index':14062464}
        tensors=payload['model']; seen=set(); tensor_count=0
        for tensor in tensors.values():
            assert tensor.dtype==torch.float32 and torch.isfinite(tensor).all()
            key=(tensor.untyped_storage().data_ptr(),tensor.storage_offset(),tuple(tensor.shape))
            if key not in seen:
                tensor_count+=tensor.numel();seen.add(key)
        assert tensor_count==49860480, tensor_count
    record={'schema_version':1,'kind':'submission evidence extract; not a new evaluation',
        'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'sources':sources,'summary':summary,'manifest':manifest,'runtime':load(paths['runtime']),
        'selection':{'selected_step':219726,'terminal_is_lowest_late_combined':True,
            'suite_selection_before_benchmark_exposure':sequence['selection_before_benchmark_exposure'],
            'preflight_mtime_utc':datetime.fromtimestamp(paths['preflight'].stat().st_mtime,timezone.utc).isoformat(),
            'first_benchmark_started_at':results['hellaswag']['metadata']['started_at'],
            'timing_caveat':'Preflight filesystem timestamp corroborates sequence; it is not a signed independent timestamp.'},
        'suite':suite,'official_results':results,'validation':validation,'train_loss_bins':bins,
        'metrics_audit':events,'checkpoint_unique_tensor_elements':tensor_count,
        'full_stream_rehashed':args.full_stream,
        'historical_exp012':load(ROOT/'results/exp012-official-provenance.json')}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print('PASS: submission evidence export; no benchmark/model execution', flush=True)


if __name__=='__main__':
    main()
