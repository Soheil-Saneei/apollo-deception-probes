"""Mirror Qwen probe evaluation on saved Llama report means without updating Llama."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import random
import sys
import warnings

import numpy as np
from safetensors import safe_open
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'apollo_roleplaying'))
from sanity_checks import C_GRID, SEEDS, fit_text, metrics, paired_bootstrap, select_classifier, sha


def evaluation_rows(rows):
    if len({r['raw_index'] for r in rows}) != len(rows):
        raise ValueError('Expected unique trajectory IDs')
    if Counter(r['label'] for r in rows) != {'honest':666, 'deceptive':334}:
        raise ValueError('Unexpected cohort labels')
    if Counter(r['split'] for r in rows) != {'train':800, 'test':200}:
        raise ValueError('Unexpected source split')
    rng = random.Random(42)
    validation = set()
    for label in ('honest', 'deceptive'):
        groups = sorted(r['raw_index'] for r in rows if r['split']=='train' and r['label']==label)
        rng.shuffle(groups)
        validation.update(groups[:round(len(groups)*150/800)])
    result = []
    for row in rows:
        split = 'validation' if row['raw_index'] in validation else row['split']
        content = row['messages'][-1]['content']
        offset = row['char_spans']['report'][0]-row['char_spans']['assistant'][0]
        text = content[offset:]
        if not text.startswith('Executed trade update'):
            raise ValueError('Unexpected report start')
        result.append(dict(trajectory_id=row['trajectory_id'], raw_index=row['raw_index'],
            apollo_loader_id=row['apollo_loader_id'], group_id=row['group_id'],
            label=row['label'], report_label=row['report_label'], original_split=row['split'],
            split=split, report_text=text, report_token_count=len(row['token_positions']['report'])))
    assert {r['raw_index'] for r in result if r['split']=='test'} == {
        r['raw_index'] for r in rows if r['split']=='test'}
    assert len(validation)==150
    group_sets = [{r['group_id'] for r in result if r['split']==s} for s in ('train','validation','test')]
    assert all(not group_sets[i] & group_sets[j] for i,j in ((0,1),(0,2),(1,2)))
    return result


def fit_layer(layer, x, masks, y, permutations, max_iter=2000):
    """Each layer is independent; use the unchanged Qwen fitter inside each job."""
    scaler = StandardScaler().fit(x[masks['train']])
    tx = scaler.transform(x[masks['train']])
    vx = scaler.transform(x[masks['validation']])
    ex = scaler.transform(x[masks['test']])
    results, predictions = [], []
    coefficient = intercept = None
    for j, train_y in enumerate([y[masks['train']]]+permutations):
        auc,c,clf = select_classifier(tx,train_y,vx,y[masks['validation']],max_iter=max_iter)
        p = clf.predict_proba(ex)[:,1]
        row = dict(layer=layer,C=c,validation_auc=float(auc)) | metrics(y[masks['test']],p)
        row['selected_fit_iterations'] = int(clf.n_iter_[0])
        if j:
            row['seed'] = SEEDS[j-1]
        else:
            coefficient, intercept = clf.coef_[0], clf.intercept_[0]
        results.append(row); predictions.append(p)
    return layer, results, predictions, coefficient, intercept, scaler.mean_, scaler.scale_


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen-metrics', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    warnings.simplefilter('error', ConvergenceWarning)
    original_rows = [json.loads(x) for x in (args.run/'manifest.jsonl').read_text().splitlines()]
    rows = evaluation_rows(original_rows)
    (args.output/'evaluation_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    masks = {s:np.array([r['split']==s for r in rows]) for s in ('train','validation','test')}
    y = np.array([int(r['label']=='deceptive') for r in rows])
    test_y = y[masks['test']]
    texts = np.array([r['report_text'] for r in rows], dtype=object)
    qwen = json.loads(args.qwen_metrics.read_text())
    with safe_open(str(args.run/'features.safetensors'), framework='numpy') as f:
        assert f.get_slice('report_mean').get_shape()==[1000,80,8192]
        assert np.array_equal(f.get_tensor('labels'), y)
        assert np.array_equal(f.get_tensor('raw_indices'), [r['raw_index'] for r in rows])
        assert np.array_equal(f.get_tensor('train_mask'), [r['original_split']=='train' for r in rows])
        features = f.get_tensor('report_mean')
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite report mean features')
    vectorizer, text_clf, text_p, text_settings = fit_text(texts, masks, y)
    import joblib
    joblib.dump(dict(vectorizer=vectorizer, classifier=text_clf), args.output/'text_classifier.joblib')
    print(json.dumps(dict(text=metrics(test_y,text_p), settings=text_settings)), flush=True)

    permutations = [np.random.default_rng(seed).permutation(y[masks['train']]) for seed in SEEDS]
    best = [None]*6
    layers = [[] for _ in range(6)]
    predictions = np.empty((6,80,len(test_y)))
    coefficients, intercepts, centers, scales = [], [], [], []
    completed = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(fit_layer,layer,features[:,layer],masks,y,permutations) for layer in range(80)]
        for future in as_completed(futures):
            out = future.result()
            completed[out[0]] = out
            if len(completed)%5==0 or len(completed)==80:
                print(json.dumps(dict(layers_completed=len(completed),total=80)),flush=True)
    # Restore layer order before selection so first-layer tie-breaking is unchanged.
    for layer in range(80):
        _, layer_results, layer_predictions, coefficient, intercept, center, scale = completed[layer]
        for j,row in enumerate(layer_results):
            predictions[j,layer] = layer_predictions[j]
            layers[j].append(row)
            if best[j] is None or row['validation_auc'] > best[j]['validation_auc']:
                best[j] = row
        coefficients.append(coefficient); intercepts.append(intercept)
        centers.append(center); scales.append(scale)
    activation_p = predictions[0,best[0]['layer']]
    length = np.array([[r['report_token_count']] for r in rows])
    scaler = StandardScaler().fit(length[masks['train']])
    clf = LogisticRegression(C=1.,max_iter=2000).fit(scaler.transform(length[masks['train']]),y[masks['train']])
    length_p = clf.predict_proba(scaler.transform(length[masks['test']]))[:,1]
    test_rows = [r for r in rows if r['split']=='test']
    difference = paired_bootstrap(test_y,activation_p,text_p,np.array([r['group_id'] for r in test_rows]))
    difference = {k:dict(activation_minus_text=v['activation_minus_text'],
        paired_trajectory_bootstrap_95_ci=v['paired_scenario_bootstrap_95_ci']) for k,v in difference.items()}
    result = dict(
        provenance=dict(manifest_sha256=sha(args.run/'manifest.jsonl'),
            source_config_sha256=sha(args.run/'config.json'), qwen_metrics_sha256=sha(args.qwen_metrics),
            script_sha256=sha(__file__), shared_qwen_helper_sha256=sha(Path(__file__).resolve().parents[1]/'apollo_roleplaying/sanity_checks.py')),
        protocol=dict(feature='report_mean', model_frozen=True, split_seed=42,
            independent_layer_workers=args.workers,
            split_counts=dict(Counter(r['split']+'/'+r['label'] for r in rows)),
            original_test_ids_unchanged=True, validation_from_original_training_only=True,
            positive_label='deceptive', auprc='average precision', accuracy_threshold=.5,
            C_grid=C_GRID, shuffled_seeds=SEEDS, shuffled_training_labels_only=True,
            scaler_vocabulary_and_idf_fitted_on_training_only=True,
            selection='C per layer and layer selected by true validation AUROC only; first candidate wins ties',
            text_input='exact report substring from Executed trade update through assistant content end',
            text_settings='TF-IDF word unigrams/bigrams, min_df=2, sublinear_tf=True',
            held_out_unit='trajectory from the same insider-trading setting',
            packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','scikit-learn','safetensors')}),
        activation=best[0], text_only=text_settings | metrics(test_y,text_p),
        shuffled_activation_primary=best[1], shuffled_activation_seeds=best[1:],
        shuffled_activation_summary={k:dict(mean=float(np.mean([r[k] for r in best[1:]])),
            sd=float(np.std([r[k] for r in best[1:]],ddof=1))) for k in metrics(test_y,text_p)},
        length_only=metrics(test_y,length_p),
        constant_baseline=metrics(test_y,np.full(len(test_y), y[masks['train']].mean())),
        activation_minus_text=difference, layers=layers[0], shuffled_layers=layers[1:])
    np.savez(args.output/'predictions.npz',test_ids=np.array([r['trajectory_id'] for r in test_rows]),
        labels=test_y, activation=activation_p,text_only=text_p,length_only=length_p,
        all_layer_probabilities=predictions,shuffled_training_labels=permutations,
        training_ids=np.array([r['trajectory_id'] for r in rows if r['split']=='train']))
    np.savez(args.output/'probes.npz',coefficients=coefficients,intercepts=intercepts,
        scaler_mean=centers,scaler_scale=scales)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    comparison=[]
    for name,key in [('Activation probe','activation'),('TF-IDF + logistic regression','text_only'),
                     ('Shuffled activation (five-seed mean)','shuffled_activation_summary'),
                     ('Length only','length_only'),('Constant/majority','constant_baseline')]:
        values=[]
        for d in (qwen,result):
            values.append({m: d[key][m]['mean'] if key=='shuffled_activation_summary' else d[key][m]
                           for m in ('auroc','auprc','accuracy')})
        comparison.append(dict(model=name,qwen=values[0],llama=values[1]))
    (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    lines=['# Qwen / Llama probe evaluation','',
        '| Model | Qwen AUROC | Qwen AUPRC | Qwen accuracy | Llama AUROC | Llama AUPRC | Llama accuracy |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in comparison:
        values=[row[d][m] for d in ('qwen','llama') for m in ('auroc','auprc','accuracy')]
        lines.append('| '+row['model']+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
    lines += ['', 'AUPRC is average precision; deceptive is positive; accuracy threshold is 0.5.',
        'Qwen: 259/56/56 scenarios (train/validation/test), 112 test completions, 50% deceptive.',
        'Llama: 650/150/200 trajectories, 33.5% deceptive on test; original test IDs unchanged.',
        'Qwen holds out roleplaying scenarios. Llama holds out trajectories from the same insider-trading setting.',
        'This is not a controlled comparison of model capability: dataset, labels, context, and class prevalence differ.',
        'Validation chooses C and layer; test metrics are not used for model selection.',
        'Shuffled controls permute training labels only, with true validation/test labels, across seeds 42-46.']
    (args.output/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('layers','shuffled_layers','provenance')},indent=2))


if __name__=='__main__':
    main()
