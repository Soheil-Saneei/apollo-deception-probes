"""Identical-cohort Qwen/Llama insider-trading probes and shared text controls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import warnings

import numpy as np
from safetensors import safe_open
from sklearn.exceptions import ConvergenceWarning
from analyze_report_controls import evaluate
from sanity_checks import paired_bootstrap, metrics, sha


def contrast(y,a,b,groups):
    return {k:dict(difference=v['activation_minus_text'],
        paired_trajectory_bootstrap_95_ci=v['paired_scenario_bootstrap_95_ci'])
        for k,v in paired_bootstrap(y,a,b,groups).items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('qwen_run',type=Path)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    warnings.simplefilter('error',ConvergenceWarning)
    assert json.loads((args.qwen_run/'summary.json').read_text())['completed']==1000
    qrows=[json.loads(s) for s in (args.qwen_run/'manifest.jsonl').read_text().splitlines()]
    positions={r['trajectory_id']:i for i,r in enumerate(qrows)}
    args.output.mkdir(parents=True,exist_ok=False)
    specs=[('report','report_mean','llama33-probe-eval-v3','evaluation_manifest.jsonl','qwen_report_token_count'),
        ('llama_length_matched','report_mean','llama33-report-controls-v2/length_matched','manifest.jsonl','qwen_report_token_count'),
        ('pre_report','report_boundary','llama33-report-controls-v2/pre_report','manifest.jsonl','qwen_pre_report_token_count'),
        ('early_reasoning','early_mean','llama33-early-reasoning-v2/probe','manifest.jsonl','qwen_early_token_count')]
    comparisons={}; lengths={}
    for name,feature,reference,manifest,length_key in specs:
        ref=args.references/reference
        rows=[json.loads(s) for s in (ref/manifest).read_text().splitlines()]
        old=json.loads((ref/'metrics.json').read_text())
        source_predictions=np.load(ref/'predictions.npz')
        selected=[]
        for row in rows:
            q=qrows[positions[row['trajectory_id']]]
            assert all(q[k]==row[k] for k in ('raw_index','group_id','label','split'))
            text=row.get('baseline_text',row['report_text'])
            selected.append(dict(row,baseline_text=text,baseline_token_count=q[length_key],
                llama_baseline_token_count=row.get('baseline_token_count',row['report_token_count']),
                qwen_activation_path=q['activation_path']))
        with safe_open(str(args.qwen_run/'features.safetensors'),framework='numpy') as f:
            assert np.array_equal(f.get_tensor('raw_indices'),[r['raw_index'] for r in qrows])
            x=f.get_tensor(feature)[[positions[r['trajectory_id']] for r in rows]]
        result=evaluate(selected,x,args.output/name,feature,args.workers,max_iter=10000)
        pred=np.load(args.output/name/'predictions.npz')
        assert np.array_equal(pred['test_ids'],source_predictions['test_ids'])
        assert np.array_equal(pred['training_ids'],source_predictions['training_ids'])
        assert np.array_equal(pred['shuffled_training_labels'],source_predictions['shuffled_training_labels'])
        text_error=float(np.max(np.abs(pred['text_only']-source_predictions['text_only'])))
        assert text_error<1e-6, 'Shared text baseline changed'
        groups=np.array([r['group_id'] for r in selected if r['split']=='test'])
        comparisons[name]=dict(llama=old,qwen=result,shared_text_max_prediction_error=text_error,
            qwen_minus_llama_activation=contrast(pred['labels'],pred['activation'],source_predictions['activation'],groups))
        lengths[name]={s:{label:dict(Counter(r['baseline_token_count'] for r in selected if r['split']==s and r['label']==label))
                         for label in ('honest','deceptive')} for s in ('train','validation','test')}
        print(json.dumps(dict(completed_analysis=name,qwen=result['activation'],llama=old['activation'])),flush=True)
    early=args.output/'early_reasoning'
    pred=np.load(early/'predictions.npz')
    full=args.references/'llama33-full-prefix-v1'
    full_pred=np.load(full/'predictions.npz')
    assert np.array_equal(pred['test_ids'],full_pred['test_ids'])
    erows=[json.loads(s) for s in (early/'manifest.jsonl').read_text().splitlines()]
    groups=np.array([r['group_id'] for r in erows if r['split']=='test'])
    full_comparison=dict(shared_full_prefix_text=json.loads((full/'metrics.json').read_text())['full_prefix_text'],
        qwen_selected_activation=comparisons['early_reasoning']['qwen']['activation'],
        llama_selected_activation=comparisons['early_reasoning']['llama']['activation'],
        qwen_layer0=metrics(pred['labels'],pred['all_layer_probabilities'][0,0]),
        qwen_layer17=metrics(pred['labels'],pred['all_layer_probabilities'][0,17]),
        qwen_selected_minus_full_prefix=contrast(pred['labels'],pred['activation'],full_pred['full_prefix_text'],groups))
    (args.output/'early_full_prefix.json').write_text(json.dumps(full_comparison,indent=2))
    (args.output/'comparison.json').write_text(json.dumps(comparisons,indent=2))
    (args.output/'qwen_length_distributions.json').write_text(json.dumps(lengths,indent=2))
    (args.output/'provenance.json').write_text(json.dumps(dict(qwen_manifest_sha256=sha(args.qwen_run/'manifest.jsonl'),
        script_sha256=sha(__file__),all_text_baselines_shared=True,all_cohorts_and_splits_unchanged=True,
        qwen_is_generating=False,activation_max_iter=10000),indent=2))
    lines=['# Frozen Qwen processing released Llama insider-trading trajectories','',
        'Qwen teacher-forces the saved text; it does not generate these trajectories.',
        'All cohorts, labels, train/validation/test assignments and text-baseline inputs match Llama.',
        'Native tokenizers/templates and model architectures differ. Both replays use FP16.',
        'The length-matched cohort was selected using Llama token counts. Qwen lengths need not match.',
        'Early windows match exact characters, so Qwen window token counts need not equal eight.','']
    for name,c in comparisons.items():
        lines += ['## '+name,'',
            '| Method | Llama AUROC | AUPRC | Accuracy | Qwen AUROC | AUPRC | Accuracy |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for label,key in [('Activation','activation'),('Shared TF-IDF','text_only'),
            ('Shuffled, five-seed mean','shuffled_activation_summary'),('Own-tokenizer length','length_only'),('Constant','constant_baseline')]:
            vals=[c[model][key][m]['mean'] if key=='shuffled_activation_summary' else c[model][key][m]
                  for model in ('llama','qwen') for m in ('auroc','auprc','accuracy')]
            lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
        lines += ['',f"Validation-selected layers (zero-based): Llama {c['llama']['activation']['layer']}, Qwen {c['qwen']['activation']['layer']}.",'']
    (args.output/'COMPARISON.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(12,13))
    for axs,(name,c) in zip(axes,comparisons.items()):
        for ax,model in zip(axs,('llama','qwen')):
            r=c[model]
            ax.plot([v['layer'] for v in r['layers']],[v['auroc'] for v in r['layers']],label='Test AUROC')
            ax.plot([v['layer'] for v in r['layers']],[v['validation_auc'] for v in r['layers']],label='Validation AUROC')
            ax.axhline(r['text_only']['auroc'],color='green',linestyle='--',label='Shared TF-IDF')
            ax.axhline(r['length_only']['auroc'],color='gray',linestyle='--',label='Own-tokenizer length')
            ax.axvline(r['activation']['layer'],color='black',linestyle=':',label='Selected layer')
            ax.set(title=name+' / '+model,xlabel='Decoder block (zero-based)',ylabel='AUROC',ylim=(.4,1.03))
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(args.output/'layerwise_comparison.png',dpi=180)
    print('ALL COMPARISONS COMPLETE',flush=True)


if __name__=='__main__':
    main()
