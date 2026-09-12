"""Surface-text and shuffled-training-label controls on the unchanged Qwen split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from safetensors.numpy import load_file
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

C_GRID = (.001, .01, .1, 1., 10.)
SEEDS = (42, 43, 44, 45, 46)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, p):
    return dict(auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                accuracy=float(accuracy_score(y, p >= .5)))


def validate_rows(rows):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate example IDs')
    groups = {}
    for r in rows:
        if hashlib.sha256(r['completion'].encode()).hexdigest() != r['completion_sha256']:
            raise ValueError('Saved completion text hash mismatch')
        if r['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        previous = groups.setdefault(r['scenario_id'], r['split'])
        if previous != r['split']:
            raise ValueError('Scenario crosses splits')
    for group in groups:
        pair = [r for r in rows if r['scenario_id'] == group]
        if Counter(r['label'] for r in pair) != {'honest': 1, 'deceptive': 1}:
            raise ValueError('Expected one honest/deceptive pair per scenario')
        if pair[0]['messages'] != pair[1]['messages']:
            raise ValueError('Paired prompts differ')
    return {s: np.array([r['split'] == s for r in rows]) for s in ('train', 'validation', 'test')}


def select_classifier(train_x, train_y, validation_x, validation_y, max_iter=2000):
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=max_iter, random_state=42)
        clf.fit(train_x, train_y)
        auc = roc_auc_score(validation_y, clf.predict_proba(validation_x)[:, 1])
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    return best


def fit_text(texts, masks, y):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(texts[masks['train']])
    x_validation = vectorizer.transform(texts[masks['validation']])
    auc, c, clf = select_classifier(x_train, y[masks['train']],
                                    x_validation, y[masks['validation']])
    p = clf.predict_proba(vectorizer.transform(texts[masks['test']]))[:, 1]
    return vectorizer, clf, p, dict(C=c, validation_auc=float(auc),
                                   vocabulary_size=len(vectorizer.vocabulary_))


def paired_bootstrap(y, a, b, groups, n=2000):
    """Resample whole held-out pairs, preserving dependence and label balance."""
    rng = np.random.default_rng(2026)
    unique = sorted(set(groups))
    indices = [np.flatnonzero(groups == g) for g in unique]
    differences = {k: [] for k in metrics(y, a)}
    for _ in range(n):
        ids = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
        ma, mb = metrics(y[ids], a[ids]), metrics(y[ids], b[ids])
        for key in differences:
            differences[key].append(ma[key]-mb[key])
    ma, mb = metrics(y, a), metrics(y, b)
    return {k: dict(activation_minus_text=ma[k]-mb[k],
                    paired_scenario_bootstrap_95_ci=np.quantile(v, [.025, .975]).tolist())
            for k, v in differences.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    warnings.simplefilter('error', ConvergenceWarning)
    manifest = args.run/'manifest.jsonl'
    rows = [json.loads(x) for x in manifest.read_text().splitlines()]
    masks = validate_rows(rows)
    y = np.array([r['label'] == 'deceptive' for r in rows], dtype=int)
    test_y = y[masks['test']]
    texts = np.array([r['completion'] for r in rows], dtype=object)
    test_rows = [r for r in rows if r['split'] == 'test']
    saved = np.load(args.run/'probes.npz')
    assert saved['test_ids'].tolist() == [r['id'] for r in test_rows]
    original = json.loads((args.run/'probe_metrics.json').read_text())
    chosen = original['selected_by_validation']
    assert chosen == max(original['layers'], key=lambda x: x['validation_auc'])
    activation_p = saved['test_probability'][chosen['layer']]
    features = load_file(str(args.run/'features.safetensors'))['mean_resid_post']
    assert len(features) == len(rows)
    # Check that saved weights and features reconstruct the original predictions.
    z = (features[masks['test'], chosen['layer']].astype(float)
         - saved['scaler_mean'][chosen['layer']])/saved['scaler_scale'][chosen['layer']]
    reconstructed = 1/(1+np.exp(-(z @ saved['coefficients'][chosen['layer']]
                                 + saved['intercepts'][chosen['layer']])))
    error = float(np.max(np.abs(reconstructed-activation_p)))
    assert error < 1e-6
    vectorizer, text_clf, text_p, text_settings = fit_text(texts, masks, y)
    import joblib
    joblib.dump(dict(vectorizer=vectorizer, classifier=text_clf), args.output/'text_classifier.joblib')
    print(json.dumps(dict(text=metrics(test_y, text_p), settings=text_settings)), flush=True)

    # Labels are shuffled inside the training split, preserving its class counts.
    # Validation/test retain true labels. Repeat the original layer/C selection.
    permutations = [np.random.default_rng(seed).permutation(y[masks['train']]) for seed in SEEDS]
    null_best = [None]*len(SEEDS)
    null_layers = [[] for _ in SEEDS]
    null_predictions = np.empty((len(SEEDS), features.shape[1], len(test_y)))
    for layer in range(features.shape[1]):
        scaler = StandardScaler().fit(features[masks['train'], layer])
        tx = scaler.transform(features[masks['train'], layer])
        vx = scaler.transform(features[masks['validation'], layer])
        ex = scaler.transform(features[masks['test'], layer])
        for j, seed in enumerate(SEEDS):
            auc, c, clf = select_classifier(tx, permutations[j], vx, y[masks['validation']])
            pp = clf.predict_proba(ex)[:, 1]
            null_predictions[j, layer] = pp
            settings = dict(seed=seed, layer=layer, C=c, validation_auc=float(auc))
            null_layers[j].append(settings | metrics(test_y, pp))
            if null_best[j] is None or auc > null_best[j]['validation_auc']:
                null_best[j] = settings | metrics(test_y, pp)
        if layer % 4 == 0 or layer == features.shape[1]-1:
            print(json.dumps(dict(shuffled_layers_completed=layer+1, total=features.shape[1])), flush=True)

    length = np.array([[len(r['completion_positions'])] for r in rows])
    scaler = StandardScaler().fit(length[masks['train']])
    clf = LogisticRegression(C=1., max_iter=2000).fit(
        scaler.transform(length[masks['train']]), y[masks['train']])
    length_p = clf.predict_proba(scaler.transform(length[masks['test']]))[:, 1]
    result = dict(
        provenance=dict(manifest_sha256=sha(manifest), original_probes_sha256=sha(args.run/'probes.npz'),
            original_metrics_sha256=sha(args.run/'probe_metrics.json'),
            script_sha256=sha(__file__), saved_probability_reconstruction_max_error=error),
        protocol=dict(positive_label='deceptive', auprc='average precision (not trapezoidal PR area)',
            accuracy_threshold=.5, unchanged_splits=True, C_grid=C_GRID,
            shuffled_training_labels_only=True, shuffled_seeds=SEEDS,
            layer_and_C_selection='true validation AUROC; test labels never used for selection',
            text_input='exact saved completion string; TF-IDF word unigrams/bigrams, min_df=2, sublinear_tf=True; training-only vocabulary and IDF',
            split_examples={s:int(m.sum()) for s,m in masks.items()},
            test_scenarios=len(set(r['scenario_id'] for r in test_rows))),
        activation=chosen | metrics(test_y, activation_p),
        text_only=text_settings | metrics(test_y, text_p),
        shuffled_activation_primary=null_best[0],
        shuffled_activation_seeds=null_best,
        shuffled_activation_summary={k:dict(mean=float(np.mean([r[k] for r in null_best])),
            sd=float(np.std([r[k] for r in null_best], ddof=1))) for k in metrics(test_y, text_p)},
        length_only=metrics(test_y, length_p),
        constant_baseline=metrics(test_y, np.full(len(test_y), .5)),
        activation_minus_text=paired_bootstrap(test_y, activation_p, text_p,
            np.array([r['scenario_id'] for r in test_rows])),
        test_error_overlap=dict(activation_only_correct=int(np.sum((activation_p>=.5)==test_y)
            - np.sum(((activation_p>=.5)==test_y) & ((text_p>=.5)==test_y))),
            text_only_correct=int(np.sum(((text_p>=.5)==test_y) & ((activation_p>=.5)!=test_y)))),
        shuffled_layers=null_layers)
    np.savez(args.output/'predictions.npz', test_ids=np.array([r['id'] for r in test_rows]),
        test_scenario_ids=np.array([r['scenario_id'] for r in test_rows]), labels=test_y,
        activation=activation_p, text_only=text_p, length_only=length_p,
        shuffled_all_layers=null_predictions, shuffled_training_labels=permutations,
        training_ids=np.array([r['id'] for r in rows if r['split']=='train']))
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Qwen roleplaying sanity checks', '',
             'Fixed split: 259 train / 56 validation / 56 test scenarios. Deceptive is positive.',
             'AUPRC is average precision. Layer and C selection use validation AUROC only.', '',
             '| Model | Test AUROC | Test AUPRC | Test accuracy |',
             '| --- | ---: | ---: | ---: |']
    for name, key in [('Activation', 'activation'), ('TF-IDF text', 'text_only'),
                      ('Shuffled activation (seed 42)', 'shuffled_activation_primary'),
                      ('Length only', 'length_only'), ('Constant', 'constant_baseline')]:
        r = result[key]
        lines.append(f"| {name} | {r['auroc']:.4f} | {r['auprc']:.4f} | {r['accuracy']:.4f} |")
    lines += ['', 'Activation minus text, paired-scenario bootstrap 95% intervals:']
    for k, v in result['activation_minus_text'].items():
        lo, hi = v['paired_scenario_bootstrap_95_ci']
        lines.append(f"- {k}: {v['activation_minus_text']:.4f} [{lo:.4f}, {hi:.4f}]")
    lines += ['', 'Shuffled-label results across five seeds (mean +/- sample SD):']
    for k, v in result['shuffled_activation_summary'].items():
        lines.append(f"- {k}: {v['mean']:.4f} +/- {v['sd']:.4f}")
    (args.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('shuffled_layers', 'provenance')}, indent=2))


if __name__ == '__main__':
    main()
