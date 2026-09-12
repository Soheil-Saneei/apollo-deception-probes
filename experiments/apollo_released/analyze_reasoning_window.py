"""Training-defined early reasoning divergence with identical activation/text windows."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
from safetensors import safe_open
from evaluate_reports import evaluation_rows
from analyze_report_controls import evaluate
from sanity_checks import sha, paired_bootstrap

WINDOW_TOKENS = 8
# Conservative explicit disclosure/concealment/sensitivity cues; fixed before fitting.
CUES = re.compile(r'\b(?:insider|sensitive|confidential|conceal\w*|hid(?:e|ing|den)|reveal\w*|disclos\w*|'
                  r'withhold\w*|omit\w*|cautious\w*|careful\w*|avoid\w*|suspicio\w*|risk\w*|'
                  r'deceiv\w*|decept\w*|illegal\w*|unethical\w*|secret\w*|source|'
                  r'not\s+(?:mention|tell|share)|frame\w*|downplay\w*)\b', re.I)


def modal_divergence(sequences, labels):
    """First per-position class mode disagreement, using training sequences only."""
    audit=[]
    for position in range(min(map(len,sequences))):
        counts={label:Counter(s[position] for s,y in zip(sequences,labels) if y==label)
                for label in ('honest','deceptive')}
        modes={label:sorted(c,key=lambda token:(-c[token],token))[0] for label,c in counts.items()}
        audit.append(dict(position=position,counts={k:dict(v) for k,v in counts.items()},modes=modes))
        if modes['honest']!=modes['deceptive']:
            return position,audit
    raise ValueError('No training modal divergence')


def select_window(offsets, body, reasoning_start, divergence):
    content=[i for i,(a,b) in enumerate(offsets) if b>reasoning_start]
    if not content or offsets[content[0]][0]<reasoning_start:
        raise ValueError('Reasoning opening crosses a token boundary')
    # Start strictly AFTER the differing token; keep exactly eight subsequent tokens.
    start=content[0]+divergence+1
    stop=start+WINDOW_TOKENS
    if stop>len(offsets):
        return None,'too_short'
    reasoning_end=body.find('"""',reasoning_start)
    if reasoning_end<0:
        return None,'missing_reasoning_close'
    cue=CUES.search(body,reasoning_start,reasoning_end)
    safe_end=min(reasoning_end,cue.start() if cue else reasoning_end)
    if offsets[stop-1][1]>safe_end:
        return None,'window_reaches_reasoning_end_or_explicit_cue'
    return (start,stop),None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--existing-evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    from tokenizers import Tokenizer
    import torch
    from safetensors.torch import save_file
    original=[json.loads(s) for s in (args.run/'manifest.jsonl').read_text().splitlines()]
    rows=evaluation_rows(original)
    existing=[json.loads(s) for s in (args.existing_evaluation/'evaluation_manifest.jsonl').read_text().splitlines()]
    assert rows==existing
    config=json.loads((args.run/'config.json').read_text())
    tp=Path('.hf-cache-llama')/('models--'+config['tokenizer'].replace('/','--'))/'snapshots'/config['tokenizer_revision']/'tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tp))
    encoded=[]
    for source in original:
        body=source['messages'][-1]['content']
        assert body.startswith('Reasoning: """')
        enc=tokenizer.encode(body,add_special_tokens=False)
        assert enc.ids==[source['input_ids'][i] for i in source['token_positions']['assistant']]
        assert tokenizer.decode(enc.ids,skip_special_tokens=False)==body
        start=len('Reasoning: """')
        content=[i for i,(a,b) in enumerate(enc.offsets) if b>start]
        assert enc.offsets[content[0]][0]==start
        encoded.append((enc,start,content[0]))
    train=[i for i,r in enumerate(rows) if r['split']=='train']
    sequences=[encoded[i][0].ids[encoded[i][2]:] for i in train]
    labels=[rows[i]['label'] for i in train]
    divergence,audit=modal_divergence(sequences,labels)
    literal=next(i for i in range(divergence+1) if
        {s[i] for s,y in zip(sequences,labels) if y=='honest'} !=
        {s[i] for s,y in zip(sequences,labels) if y=='deceptive'})
    for a in audit:
        a['decoded_modes']={k:tokenizer.decode([v],skip_special_tokens=False) for k,v in a['modes'].items()}
        a['decoded_counts']={label:{tokenizer.decode([int(k)],skip_special_tokens=False):v for k,v in counts.items()}
                             for label,counts in a['counts'].items()}
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'token_activations').mkdir()
    protocol=dict(source_manifest_sha256=sha(args.run/'manifest.jsonl'),source_config_sha256=sha(args.run/'config.json'),
        existing_split_sha256=sha(args.existing_evaluation/'evaluation_manifest.jsonl'),script_sha256=sha(__file__),
        model_frozen=True,feature_source='original FP16 frozen causal replay; no model update',
        divergence_definition='earliest training-only per-position modal token disagreement between labels',
        literal_support_difference_position=literal,divergence_reasoning_token_zero_based=divergence,
        window_start_reasoning_token_zero_based=divergence+1,window_tokens=WINDOW_TOKENS,
        pooling='FP32 mean over exactly eight post-divergence reasoning-token block outputs',
        activation_max_iter=10000,iteration_change_reason='2000-iteration shuffled fit failed to converge',
        text='exact decoded same eight tokens, including original whitespace; no earlier/later text',
        fixed_exclusion_regex=CUES.pattern,paired_counterfactuals=False,
        caveat='Activations attend earlier context; TF-IDF only sees the specified window.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (args.output/'divergence_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(divergence=divergence,audit=audit)),flush=True)
    selected,excluded,means=[],[],[]
    for i,(source,row,(enc,start,content_start)) in enumerate(zip(original,rows,encoded)):
        body=source['messages'][-1]['content']
        window,reason=select_window(enc.offsets,body,start,divergence)
        if window is None:
            excluded.append(dict(trajectory_id=row['trajectory_id'],split=row['split'],label=row['label'],reason=reason))
            continue
        a,b=window
        text=tokenizer.decode(enc.ids[a:b],skip_special_tokens=False)
        assert text==body[enc.offsets[a][0]:enc.offsets[b-1][1]]
        assert not CUES.search(text)
        with safe_open(str(args.run/source['activation_path']),framework='pt') as f:
            assert f.metadata()['trajectory_id']==row['trajectory_id']
            values=f.get_slice('resid_post')[:,a:b,:].contiguous()
        assert list(values.shape)==[80,WINDOW_TOKENS,8192] and torch.isfinite(values).all()
        path='token_activations/'+row['trajectory_id']+'.safetensors'
        save_file(dict(resid_post=values),str(args.output/path),metadata=dict(trajectory_id=row['trajectory_id'],label=row['label'],split=row['split']))
        means.append(values.float().mean(1).numpy())
        selected.append(dict(row,baseline_text=text,baseline_token_count=WINDOW_TOKENS,
            window_token_ids=enc.ids[a:b],window_assistant_token_range=[a,b],
            window_absolute_token_range=[source['token_positions']['assistant'][0]+a,source['token_positions']['assistant'][0]+b],
            window_char_range=[enc.offsets[a][0],enc.offsets[b-1][1]],token_activation_path=path))
        if len(selected)%100==0:
            print(json.dumps(dict(windows_saved=len(selected))),flush=True)
    (args.output/'exclusions.json').write_text(json.dumps(excluded,indent=2))
    (args.output/'window_text_audit.json').write_text(json.dumps(dict(Counter(r['baseline_text'] for r in selected)),indent=2))
    result=evaluate(selected,np.stack(means),args.output/'probe','early_reasoning_window',args.workers,max_iter=10000)
    predictions=np.load(args.output/'probe/predictions.npz')
    test_rows=[r for r in selected if r['split']=='test']
    difference=paired_bootstrap(predictions['labels'],predictions['activation'],predictions['text_only'],
        np.array([r['group_id'] for r in test_rows]))
    (args.output/'activation_minus_text.json').write_text(json.dumps(difference,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,m in zip(axes,('auroc','auprc','accuracy')):
        ax.plot([r[m] for r in result['layers']],label='Activation test')
        null=np.array([[r[m] for r in rs] for rs in result['shuffled_layers']])
        ax.plot(null.mean(0),color='gray',label='Shuffled mean')
        ax.fill_between(range(80),null.min(0),null.max(0),color='gray',alpha=.15)
        ax.axhline(result['text_only'][m],color='green',linestyle='--',label='Same-window TF-IDF')
        ax.axhline(result['constant_baseline'][m],color='orange',linestyle='--',label='Constant baseline')
        ax.axvline(result['activation']['layer'],color='black',linestyle=':',label='Validation-selected layer')
        ax.set(title=m.upper(),xlabel='Decoder block (zero-based)',ylim=(0,1.03))
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(args.output/'layerwise_window.png',dpi=180)
    print(json.dumps({k:v for k,v in result.items() if k not in ('layers','shuffled_layers')},indent=2))


if __name__=='__main__':
    main()
