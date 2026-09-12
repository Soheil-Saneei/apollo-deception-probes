"""Train-only scaling, validation selection, scenario-held-out linear probes."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    rows = [json.loads(x) for x in (args.run/"manifest.jsonl").read_text().splitlines()]
    groups = {s: {r["scenario_id"] for r in rows if r["split"] == s}
              for s in ("train", "validation", "test")}
    assert all(not groups[a] & groups[b] for a, b in
               (("train", "test"), ("train", "validation"), ("validation", "test")))
    masks = {s: np.array([r["split"] == s for r in rows]) for s in groups}
    y = np.array([r["label"] == "deceptive" for r in rows], dtype=int)
    features = load_file(str(args.run/"features.safetensors"))["mean_resid_post"]
    metrics, weights, intercepts, centers, scales, predictions = [], [], [], [], [], []
    for layer in range(features.shape[1]):
        x = features[:, layer]
        candidates = []
        for c in (.001, .01, .1, 1., 10.):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000,
                                                                      random_state=42))
            clf.fit(x[masks["train"]], y[masks["train"]])
            score = roc_auc_score(y[masks["validation"]],
                                 clf.predict_proba(x[masks["validation"]])[:, 1])
            candidates.append((score, c, clf))
        val_auc, c, clf = max(candidates, key=lambda t: t[0])
        test_probability = clf.predict_proba(x[masks["test"]])[:, 1]
        metrics.append(dict(layer=layer, C=c, validation_auc=val_auc,
            test_auc=roc_auc_score(y[masks["test"]], test_probability),
            test_accuracy=accuracy_score(y[masks["test"]], test_probability >= .5)))
        weights.append(clf[-1].coef_[0]); intercepts.append(clf[-1].intercept_[0])
        centers.append(clf[0].mean_); scales.append(clf[0].scale_)
        predictions.append(test_probability)
    selected = max(metrics, key=lambda r: r["validation_auc"])
    length = np.array([[len(r["completion_positions"])] for r in rows])
    length_clf = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
    length_clf.fit(length[masks["train"]], y[masks["train"]])
    length_p = length_clf.predict_proba(length[masks["test"]])[:, 1]
    result = dict(scenario_counts={s:len(g) for s,g in groups.items()},
        majority_accuracy=.5, selected_by_validation=selected, layers=metrics,
        completion_length_baseline=dict(test_auc=roc_auc_score(y[masks["test"]], length_p),
            test_accuracy=accuracy_score(y[masks["test"]], length_p >= .5)))
    (args.run/"probe_metrics.json").write_text(json.dumps(result, indent=2))
    np.savez(args.run/"probes.npz", coefficients=weights, intercepts=intercepts,
             scaler_mean=centers, scaler_scale=scales, test_probability=predictions,
             test_ids=np.array([r["id"] for r in rows if r["split"] == "test"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot([r["layer"] for r in metrics], [r["test_auc"] for r in metrics], label="Test AUROC")
    plt.plot([r["layer"] for r in metrics], [r["validation_auc"] for r in metrics], label="Validation AUROC")
    plt.axhline(.5, color="gray", linestyle="--")
    plt.xlabel("Decoder layer (zero-based)"); plt.ylabel("AUROC")
    plt.legend(); plt.tight_layout(); plt.savefig(args.run/"probe_by_layer.png", dpi=160)
    print(json.dumps({k:v for k,v in result.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
