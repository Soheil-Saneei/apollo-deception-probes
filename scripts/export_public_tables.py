"""Create compact text/label/split tables from unchanged saved experiment manifests."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return [json.loads(line) for line in p.read_text().splitlines()]


def write(path, rows):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def main():
    role = read('experiments/apollo_roleplaying/outputs/qwen9b-paired-v2/manifest.jsonl')
    assert len(role) == 742 and len({r['scenario_id'] for r in role}) == 371
    for scenario in {r['scenario_id'] for r in role}:
        pair = [r for r in role if r['scenario_id'] == scenario]
        assert len(pair) == 2 and {r['label'] for r in pair} == {'honest', 'deceptive'}
        assert len({r['split'] for r in pair}) == 1
    keys = ['id','source_index','scenario_id','split','label','messages','completion','completion_sha256']
    write('data/roleplaying/completions.jsonl', [{k:r[k] for k in keys} for r in role])

    reports = read('experiments/apollo_released/outputs/qwen35-llama-trading-a100-v1/manifest.jsonl')
    assert len(reports) == 1000 and Counter(r['label'] for r in reports) == {'honest':666,'deceptive':334}
    assert len({r['trajectory_id'] for r in reports}) == 1000
    keys = ['trajectory_id','raw_index','apollo_loader_id','group_id','label','report_label','original_split','split','report_text','report_token_count','messages','exact_final_assistant_sha256']
    write('data/insider_trading/reports.jsonl', [{k:r[k] for k in keys} for r in reports])
    early = read('experiments/apollo_released/outputs/llama33-early-reasoning-v2/probe/manifest.jsonl')
    prefix = {r['trajectory_id']:r for r in read('experiments/apollo_released/outputs/llama33-full-prefix-v1/manifest.jsonl')}
    assert len(early) == len(prefix) == 999
    windows = []
    for r in early:
        p = prefix[r['trajectory_id']]
        assert (r['label'],r['split']) == (p['label'],p['split'])
        windows.append(dict(trajectory_id=r['trajectory_id'],label=r['label'],split=r['split'],
                            window_text=r['baseline_text'],full_prefix_text=p['prefix_text'],
                            llama_window_token_ids=r['window_token_ids'],
                            window_char_range=r['window_char_range'],prefix_sha256=p['prefix_sha256']))
    write('data/insider_trading/early_windows.jsonl', windows)
    summary = dict(source_sha256=SOURCES, roleplaying=dict(completions=742,scenarios=371,split_completions=dict(Counter(r['split'] for r in role))),
                   insider_trading=dict(reports=1000,labels=dict(Counter(r['label'] for r in reports)),split_trajectories=dict(Counter(r['split'] for r in reports)),early_windows=999),
                   notes='Text export only; original saved labels, splits, IDs, and strings are unchanged. Token arrays and full artifacts remain in release archives.')
    (ROOT/'data/table_provenance.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
