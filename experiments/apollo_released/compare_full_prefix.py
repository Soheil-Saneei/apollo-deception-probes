"""Full-prefix text control at the frozen early-window probe's exact endpoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apollo_roleplaying'))
from sanity_checks import fit_text, metrics, paired_bootstrap, sha, C_GRID


def contrast(y,a,b,groups):
    return {key:dict(difference=value['activation_minus_text'],
        paired_trajectory_bootstrap_95_ci=value['paired_scenario_bootstrap_95_ci'])
        for key,value in paired_bootstrap(y,a,b,groups).items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('replay',type=Path)
    p.add_argument('--window-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    warnings.simplefilter('error',ConvergenceWarning)
    source={r['trajectory_id']:r for r in
        (json.loads(s) for s in (args.replay/'manifest.jsonl').read_text().splitlines())}
    rows=[json.loads(s) for s in (args.window_run/'probe/manifest.jsonl').read_text().splitlines()]
    config=json.loads((args.replay/'config.json').read_text())
    tokenizer_path=Path('.hf-cache-llama')/('models--'+config['tokenizer'].replace('/','--'))/'snapshots'/config['tokenizer_revision']/'tokenizer.json'
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    prefixes=[]
    for row in rows:
        original=source[row['trajectory_id']]
        start,stop=row['window_absolute_token_range']
        assert original['label']==row['label']
        assert original['input_ids'][start:stop]==row['window_token_ids']
        ids=original['input_ids'][:stop]
        text=tokenizer.decode(ids,skip_special_tokens=False)
        assert text.endswith(row['baseline_text'])
        assert len(text)==original['char_spans']['assistant'][0]+row['window_char_range'][1]
        assert stop<=original['boundary_positions']['report']
        prefixes.append(dict(trajectory_id=row['trajectory_id'],raw_index=row['raw_index'],
            group_id=row['group_id'],label=row['label'],split=row['split'],prefix_text=text,
            prefix_token_ids=ids,prefix_token_count=stop,prefix_sha256=hashlib.sha256(text.encode()).hexdigest()))
    assert len(prefixes)==999
    masks={s:np.array([r['split']==s for r in rows]) for s in ('train','validation','test')}
    groups={s:{r['group_id'] for r in rows if r['split']==s} for s in masks}
    assert all(not groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
    y=np.array([int(r['label']=='deceptive') for r in rows]); test_y=y[masks['test']]
    predictions=np.load(args.window_run/'probe/predictions.npz')
    assert predictions['test_ids'].tolist()==[r['trajectory_id'] for r in rows if r['split']=='test']
    assert predictions['training_ids'].tolist()==[r['trajectory_id'] for r in rows if r['split']=='train']
    assert np.array_equal(predictions['labels'],test_y)
    saved=json.loads((args.window_run/'probe/metrics.json').read_text())
    assert saved['activation']['layer']==17
    layer0=predictions['all_layer_probabilities'][0,0]
    layer17=predictions['all_layer_probabilities'][0,17]
    assert np.array_equal(layer17,predictions['activation'])
    for layer,probability in [(0,layer0),(17,layer17)]:
        for k,v in metrics(test_y,probability).items():
            assert abs(v-saved['layers'][layer][k])<1e-12
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prefixes))
    vectorizer,clf,text_p,settings=fit_text(np.array([r['prefix_text'] for r in prefixes],dtype=object),masks,y)
    import joblib
    joblib.dump(dict(vectorizer=vectorizer,classifier=clf),args.output/'text_classifier.joblib')
    result=dict(protocol=dict(examples=len(rows),counts=dict(Counter(r['split']+'/'+r['label'] for r in rows)),
        full_prefix='all saved input tokens [0,window_end), including system prompt, prior messages, reasoning wrapper and window',
        same_endpoint_and_examples=True,model_frozen=True,activation_probes_reused=True,
        text_settings='word TF-IDF unigrams/bigrams, min_df=2, sublinear TF; train-only vocabulary/IDF',
        C_grid=C_GRID,selection='validation AUROC, first C wins ties; existing layer 17 retained',
        text_solver='L2 lbfgs, max_iter=2000, random_state=42',positive_label='deceptive',
        auprc='average precision',accuracy_threshold=.5),
        provenance=dict(replay_manifest_sha256=sha(args.replay/'manifest.jsonl'),
            window_manifest_sha256=sha(args.window_run/'probe/manifest.jsonl'),
            window_predictions_sha256=sha(args.window_run/'probe/predictions.npz'),
            tokenizer_json_sha256=sha(tokenizer_path),script_sha256=sha(__file__)),
        activation_layer0=saved['layers'][0],activation_layer17=saved['layers'][17],
        full_prefix_text=settings|metrics(test_y,text_p),window_only_text=saved['text_only'])
    print(json.dumps({k:v for k,v in result.items() if k not in ('protocol','provenance')},indent=2),flush=True)
    test_groups=np.array([r['group_id'] for r in rows if r['split']=='test'])
    result['layer17_minus_full_prefix']=contrast(test_y,layer17,text_p,test_groups)
    result['layer0_minus_full_prefix']=contrast(test_y,layer0,text_p,test_groups)
    result['layer17_minus_layer0']=contrast(test_y,layer17,layer0,test_groups)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    np.savez(args.output/'predictions.npz',test_ids=predictions['test_ids'],labels=test_y,
        full_prefix_text=text_p,activation_layer0=layer0,activation_layer17=layer17,
        window_only_text=predictions['text_only'])
    print(json.dumps({k:v for k,v in result.items() if '_minus_' in k},indent=2),flush=True)


if __name__=='__main__':
    main()
