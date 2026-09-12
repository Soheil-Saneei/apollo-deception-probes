"""Length-exact report controls and causal pre-report probes on saved frozen Llama."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import random
import warnings

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from evaluate_reports import evaluation_rows, fit_layer
from sanity_checks import C_GRID, SEEDS, fit_text, metrics, sha


def match_lengths(rows, seed=42):
    """Maximal exact 1:1 matches within each fixed split, without replacement."""
    rng = random.Random(seed)
    chosen, pairs = set(), []
    for split in ('train', 'validation', 'test'):
        buckets = defaultdict(lambda: {'honest': [], 'deceptive': []})
        for i, row in enumerate(rows):
            if row['split'] == split:
                buckets[row['report_token_count']][row['label']].append(i)
        for length, groups in sorted(buckets.items()):
            for group in groups.values():
                group.sort(key=lambda i: rows[i]['raw_index'])
                rng.shuffle(group)
            for a, b in zip(groups['honest'], groups['deceptive']):
                chosen.update((a, b))
                pairs.append(dict(split=split, report_token_count=length,
                    honest_id=rows[a]['trajectory_id'], deceptive_id=rows[b]['trajectory_id']))
    indices = sorted(chosen)
    for split in ('train', 'validation', 'test'):
        counts = [Counter(rows[i]['report_token_count'] for i in indices
                  if rows[i]['split'] == split and rows[i]['label'] == label)
                  for label in ('honest', 'deceptive')]
        if not counts[0] or counts[0] != counts[1]:
            raise ValueError('Missing or unequal exact matches')
    return indices, pairs


def evaluate(rows, x, output, feature, workers, max_iter=2000):
    output.mkdir()
    assert np.isfinite(x).all()
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    y = np.array([int(r['label'] == 'deceptive') for r in rows])
    masks = {s: np.array([r['split'] == s for r in rows]) for s in ('train','validation','test')}
    save_file(dict(activations=x, labels=y, raw_indices=np.array([r['raw_index'] for r in rows])),
              str(output/'activations.safetensors'), metadata={'axes':'trajectory,layer,hidden','feature':feature})
    texts = np.array([r['baseline_text'] for r in rows], dtype=object)
    vectorizer, clf, text_p, text_settings = fit_text(texts, masks, y)
    import joblib
    joblib.dump(dict(vectorizer=vectorizer, classifier=clf), output/'text_classifier.joblib')
    permutations = [np.random.default_rng(s).permutation(y[masks['train']]) for s in SEEDS]
    completed = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=warnings.simplefilter, initargs=('error', ConvergenceWarning)) as pool:
        futures = [pool.submit(fit_layer, l, x[:,l], masks, y, permutations, max_iter) for l in range(x.shape[1])]
        for future in as_completed(futures):
            try:
                out = future.result()
            except Exception as error:
                print(f'Layer fit failed: {error}', flush=True)
                for pending in futures:
                    pending.cancel()
                raise
            completed[out[0]] = out
            if len(completed)%10 == 0:
                print(json.dumps(dict(analysis=feature,layers_completed=len(completed))), flush=True)
    layers = [[] for _ in range(6)]
    predictions = np.empty((6,x.shape[1],masks['test'].sum()))
    weights, intercepts, centers, scales = [], [], [], []
    for l in range(x.shape[1]):
        _, results, probs, w, b, center, scale = completed[l]
        for j in range(6):
            layers[j].append(results[j]); predictions[j,l] = probs[j]
        weights.append(w); intercepts.append(b); centers.append(center); scales.append(scale)
    selected = [max(rs, key=lambda r:r['validation_auc']) for rs in layers]
    length = np.array([[r['baseline_token_count']] for r in rows])
    scaler = StandardScaler().fit(length[masks['train']])
    clf = LogisticRegression(C=1.,max_iter=2000).fit(scaler.transform(length[masks['train']]),y[masks['train']])
    length_p = clf.predict_proba(scaler.transform(length[masks['test']]))[:,1]
    joblib.dump(dict(scaler=scaler,classifier=clf),output/'length_classifier.joblib')
    result = dict(feature=feature, activation_max_iter=max_iter, counts=dict(Counter(r['split']+'/'+r['label'] for r in rows)),
        activation=selected[0], text_only=text_settings | metrics(y[masks['test']],text_p),
        length_only=metrics(y[masks['test']],length_p),
        constant_baseline=metrics(y[masks['test']],np.full(masks['test'].sum(),y[masks['train']].mean())),
        shuffled_activation_primary=selected[1], shuffled_activation_seeds=selected[1:],
        shuffled_activation_summary={m:dict(mean=float(np.mean([r[m] for r in selected[1:]])),
            sd=float(np.std([r[m] for r in selected[1:]],ddof=1))) for m in ('auroc','auprc','accuracy')},
        layers=layers[0], shuffled_layers=layers[1:])
    np.savez(output/'probes.npz',coefficients=weights,intercepts=intercepts,scaler_mean=centers,scaler_scale=scales)
    np.savez(output/'predictions.npz',test_ids=np.array([r['trajectory_id'] for r in rows if r['split']=='test']),
        labels=y[masks['test']], activation=predictions[0,selected[0]['layer']],
        text_only=text_p,length_only=length_p,all_layer_probabilities=predictions,
        shuffled_training_labels=permutations,
        training_ids=np.array([r['trajectory_id'] for r in rows if r['split']=='train']))
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--existing-evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    warnings.simplefilter('error',ConvergenceWarning)
    original=[json.loads(s) for s in (args.run/'manifest.jsonl').read_text().splitlines()]
    rows=evaluation_rows(original)
    existing=[json.loads(s) for s in (args.existing_evaluation/'evaluation_manifest.jsonl').read_text().splitlines()]
    assert rows==existing, 'Existing evaluation assignments changed'
    indices,pairs=match_lengths(rows)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(source_manifest_sha256=sha(args.run/'manifest.jsonl'),
        source_config_sha256=sha(args.run/'config.json'),existing_split_sha256=sha(args.existing_evaluation/'evaluation_manifest.jsonl'),
        script_sha256=sha(__file__),matching='exact token count, within fixed splits, seed 42, without replacement',
        model_frozen=True,activations='reused from original frozen FP16 replay, no model updates',
        C_grid=C_GRID,shuffled_seeds=SEEDS,shuffled_training_labels_only=True,
        selection='validation AUROC only; smallest C and earliest layer win ties',
        scaling='training only',accuracy_threshold=.5,auprc='average precision',positive_label='deceptive',
        prereport_baselines='decoded input token prefix through report boundary; prefix token count',
        matched_baselines='exact report substring; report token count')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (args.output/'matching.json').write_text(json.dumps(dict(pairs=pairs,
        excluded_ids=[r['trajectory_id'] for i,r in enumerate(rows) if i not in set(indices)]),indent=2))
    with safe_open(str(args.run/'features.safetensors'),framework='numpy') as f:
        assert np.array_equal(f.get_tensor('raw_indices'),[r['raw_index'] for r in rows])
        assert np.array_equal(f.get_tensor('labels'),[int(r['label']=='deceptive') for r in rows])
        matched=f.get_tensor('report_mean')[indices]
    matched_rows=[dict(rows[i],baseline_text=rows[i]['report_text'],baseline_token_count=rows[i]['report_token_count']) for i in indices]
    results=[evaluate(matched_rows,matched,args.output/'length_matched','report_mean',args.workers)]
    del matched
    from tokenizers import Tokenizer
    config=json.loads((args.run/'config.json').read_text())
    tokenizer_path=Path('.hf-cache-llama')/('models--'+config['tokenizer'].replace('/','--'))/'snapshots'/config['tokenizer_revision']/'tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    pre_rows=[]
    for row, source in zip(rows,original):
        boundary=source['boundary_positions']['report']
        assert boundary==source['token_positions']['report'][0]-1
        ids=source['input_ids'][:boundary+1]
        text=tokenizer.decode(ids,skip_special_tokens=False)
        assert len(text)==source['boundary_token_offsets']['report'][0]
        pre_rows.append(dict(row,baseline_text=text,baseline_token_count=len(ids),boundary_position=boundary))
    with safe_open(str(args.run/'features.safetensors'),framework='numpy') as f:
        pre=f.get_tensor('report_boundary')
    assert pre.shape==(1000,80,8192)
    results.append(evaluate(pre_rows,pre,args.output/'pre_report','report_boundary',args.workers))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for row_axes,result,title in zip(axes,results,['Exact length-matched reports (14 test trajectories)','Pre-report (200 test trajectories)']):
        for ax,m in zip(row_axes,('auroc','auprc','accuracy')):
            ax.plot([r['layer'] for r in result['layers']],[r[m] for r in result['layers']],label='Test activation')
            null=np.array([[r[m] for r in rs] for rs in result['shuffled_layers']])
            ax.plot(null.mean(0),color='gray',label='Shuffled mean')
            ax.fill_between(range(80),null.min(0),null.max(0),color='gray',alpha=.15,label='Shuffled seed range')
            ax.axhline(result['text_only'][m],color='green',linestyle='--',label='TF-IDF')
            ax.axhline(result['length_only'][m],color='orange',linestyle='--',label='Length')
            ax.axvline(result['activation']['layer'],color='black',linestyle=':',label='Validation-selected layer')
            ax.set(title=title+'\n'+m.upper(),ylim=(0,1.03),xlabel='Decoder block (zero-based)')
    axes[0,0].legend(fontsize=7,loc='lower right');fig.tight_layout()
    fig.savefig(args.output/'layerwise_controls.png',dpi=180)
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,split in zip(axes,('train','validation','test')):
        for label in ('honest','deceptive'):
            counts=Counter(r['report_token_count'] for r in matched_rows if r['split']==split and r['label']==label)
            ax.plot(sorted(counts),[counts[k] for k in sorted(counts)],marker='o',alpha=.6,label=label)
        ax.set(title=split,xlabel='Report tokens',ylabel='Count');ax.legend()
    fig.tight_layout();fig.savefig(args.output/'matched_lengths.png',dpi=180)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('layers','shuffled_layers')} for r in results],indent=2))


if __name__=='__main__':
    main()
