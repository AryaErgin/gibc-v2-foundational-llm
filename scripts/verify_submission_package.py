"""Offline submission checks. Reads repository files only; no model or datasets."""
from __future__ import annotations
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
CURRENT=['README.md','RESULTS.md','ARCHITECTURE.md','DATA_SOURCES.md','AI_ASSISTANCE.md','PROJECT_PLAN.md','AGENTS.md','EXPERIMENT_LOG.md','SOURCE_LEDGER.md','CODE_ATTRIBUTION.md','docs/EXP020_OFFICIAL_EVALUATION_PROTOCOL.md','docs/OPERATIONAL_THERMAL_CONTROLS.md','results/EXP-020-summary.md']
HASHES={'checkpoint':'95338530dcfa660acb149c94d72c07173c14dece6de7da4a22d7aa856723ce89','config':'25f473f27a3ba673d3e32cb10845e47bf7f4abf0dd47e891db812d6871c3bace','tokenizer':'c5592fba176c3d2f7915a3812559a24d7a669206f4a22484b053c8a9ce08be14','summary':'31c795afce30b7060b58bd8aa5c8ff14aad98ccb79302c450c8f75d48a995d8a','stream':'94e39c09e7696a9668568802c37e6458799b47284efa566f74fb6792f571440e'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def evidence_errors(d):
    errors=[];s=d['summary']
    def check(ok,message):
        if not ok:errors.append(message)
    check(s['parameter_count']==d['checkpoint_unique_tensor_elements']==49860480,'parameter count')
    check(s['final_step']==219726 and s['prediction_tokens']==7199981568 and s['next_sequence_index']==14062464,'terminal arithmetic')
    check(s['status']=='FULL HORIZON COMPLETE' and not s['dry_run'],'completed training')
    check(s['git_commit']=='d88800733846c7a30e2044fa0a20f9e4a448f328','training source identity')
    check(s['inter_update_sleep_seconds']==0.3 and not s['cautious_weight_decay'],'frozen operational recipe')
    check(d['sources']['checkpoint']['bytes']==598447091,'checkpoint bytes')
    check(d['sources']['stream']['bytes']==14399963138 and d['full_stream_rehashed'],'stream audit')
    for key,value in HASHES.items():check(d['sources'][key]['sha256']==value,key+' identity')
    check(d['metrics_audit']['train_rows_verified']==219726,'training rows')
    check(d['selection']['selected_step']==219726 and d['selection']['terminal_is_lowest_late_combined'] and d['selection']['suite_selection_before_benchmark_exposure'],'selection freeze')
    check(d['suite']['state']=='succeeded' and d['suite']['returncode']==0,'official suite status')
    for r in d['validation']:
        check(r['tokens']==r['step']*32768,'validation token arithmetic')
        check(math.isclose(r['combined'],(r['general']+r['edu'])/2,abs_tol=1e-12),'combined arithmetic')
    expected={'hellaswag':('acc_norm,none',0.3016331408086039),'arc_easy':('acc_norm,none',0.39141414141414144),'piqa':('acc_norm,none',0.6044613710554951),'winogrande':('acc,none',0.4877663772691397),'wikitext103':('perplexity',31.783151406614728)}
    for task,(key,value) in expected.items():
        r=d['official_results'][task];m=r['metadata']
        check(r['metrics'][key]==value,task+' frozen score')
        check(m['device']=='cpu' and m['precision']=='fp32' and m['cuda_available'] is False and m['num_fewshot']==0 and m['batch_size']==16 and m['context_length']==512,task+' protocol')
        check(m['checkpoint_sha256']==HASHES['checkpoint'],task+' checkpoint')
    return errors


def link_errors(root):
    errors=[]
    docs=[root/p for p in CURRENT]+list((root/'docs/submission').glob('*.md'))+list((root/'paper').glob('*.md'))+[root/'docs/assets/exp020/README.md']
    for path in docs:
        if not path.is_file():errors.append('Missing document: '+str(path));continue
        text=re.sub(r'```.*?```','',path.read_text(encoding='utf-8'),flags=re.S)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=target.strip('<>')
            if re.match(r'^[a-z]+://',target) or target.startswith('#'):continue
            dest=unquote(target.split('#')[0])
            if not (path.parent/dest).exists():errors.append(f'{path.relative_to(root)}: missing {target}')
    return errors


def figure_errors(root):
    folder=root/'docs/assets/exp020';errors=[]
    p=folder/'figure-provenance.json'
    if not p.exists():return ['missing figure provenance']
    meta=json.loads(p.read_text())
    if meta['evidence_sha256']!=sha(root/'results/exp020-submission-evidence.json'):errors.append('figure evidence identity')
    if meta['renderer_sha256']!=sha(root/'scripts/render_submission_figures.py'):errors.append('renderer identity')
    for name,expected in meta['inputs'].items():
        if sha(root/name)!=expected:errors.append('figure input identity: '+name)
    for name,expected in meta['outputs'].items():
        f=folder/name
        if not f.exists() or sha(f)!=expected:errors.append('figure output identity: '+name);continue
        if f.suffix=='.png':
            b=f.read_bytes()
            if b[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',b[16:24])!=(1280,720):errors.append('PNG format: '+name)
        else:
            tree=ET.parse(f)
            if tree.getroot().attrib.get('viewBox')!='0 0 1280 720':errors.append('SVG dimensions: '+name)
    for name in ('trajectory','progression','decisions','pipeline'):
        for suffix in ('.png','.svg'):
            if name+suffix not in meta['outputs']:errors.append('missing export: '+name+suffix)
    return errors


def main():
    d=json.loads((ROOT/'results/exp020-submission-evidence.json').read_text())
    errors=evidence_errors(d)+link_errors(ROOT)+figure_errors(ROOT)
    if sha(ROOT/'configs/exp020-final-7p2b-cosine.yaml')!=HASHES['config']:errors.append('live frozen config hash')
    for error in errors:print('FAIL:',error)
    if errors:raise SystemExit(1)
    print('PASS: frozen evidence contract, config SHA, submission links, four PNG/SVG pairs and figure provenance. No external artifact rehash, network, model or benchmark execution.')


if __name__=='__main__':main()
