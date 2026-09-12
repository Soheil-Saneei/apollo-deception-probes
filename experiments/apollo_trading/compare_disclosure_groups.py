#!/usr/bin/env python3
"""Exact 4-vs-24 label-permutation test using saved report-boundary vectors only."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

NONCLEAR = [6, 21, 54, 62]


def permutation_distances(vectors, small_size):
    """Enumerate every unique group assignment. Return L2 [assignments,layers]."""
    x = vectors.double()
    n, layers, _ = x.shape
    if not 0 < small_size < n or not torch.isfinite(x).all():
        raise ValueError("invalid split or nonfinite vectors")
    assignments = list(itertools.combinations(range(n), small_size))
    weights = torch.full((len(assignments), n), -1 / (n - small_size), dtype=torch.float64)
    for row, chosen in enumerate(assignments):
        weights[row, list(chosen)] = 1 / small_size
    # Centering leaves the group difference unchanged and improves conditioning.
    centered = x - x.mean(dim=0)
    result = torch.empty(len(assignments), layers, dtype=torch.float64)
    for layer in range(layers):
        gram = centered[:, layer] @ centered[:, layer].T
        result[:, layer] = ((weights @ gram) * weights).sum(dim=1).clamp_min(0).sqrt()
    return result


def holm(p):
    order = torch.argsort(p)
    adjusted = torch.empty_like(p)
    scaled = p[order] * torch.arange(len(p), 0, -1, dtype=p.dtype)
    adjusted[order] = torch.cummax(scaled, dim=0).values.clamp(max=1)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Prior 62-vs-24 comparison directory')
    parser.add_argument('--additional', type=Path, required=True, help='Boundary vectors for 06, 21, 54')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory')
    torch.set_num_threads(2)
    source = json.loads((args.reference / 'manifest.json').read_text())
    extra = json.loads((args.additional / 'manifest.json').read_text())
    if any(source[k] != extra[k] for k in ('model', 'revision')):
        raise ValueError('model/revision mismatch')
    if set(source['disclosure_indices']) & set(NONCLEAR) or len(source['disclosure_indices']) != 24:
        raise ValueError('expected disjoint cohort of 24 disclosures')
    if {x['index'] for x in extra['alignment']} != {6,21,54}:
        raise ValueError('missing or unexpected additional cases')
    for record in extra['alignment']:
        if record['layers'] != source['layers'] or record['replay_overlap_check']['max_absolute_difference'] != 0:
            raise ValueError('extra vectors must match layer order and original overlap exactly')
    data = load_file(str(args.reference / 'comparison.safetensors'))
    small = torch.stack([load_file(str(args.additional / f'trajectory-{i:02d}-boundary.safetensors'))['resid_post'].double()
                         for i in NONCLEAR[:-1]] + [data['trajectory_62'].double()])
    large = data['disclosure_individuals'].double()
    small_mean, large_mean = small.mean(dim=0), large.mean(dim=0)
    difference = small_mean - large_mean
    l2 = difference.norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(small_mean, large_mean, dim=-1)
    null = permutation_distances(torch.cat([small, large]), len(small))
    # First lexicographic assignment is the observed first-four split.
    torch.testing.assert_close(null[0], l2, rtol=1e-9, atol=1e-9)
    tolerance = 1e-10 * l2.clamp_min(1)
    p = (null >= l2 - tolerance).double().mean(dim=0)
    corrected = holm(p)
    quantiles = torch.quantile(null, torch.tensor([.5,.95,.99], dtype=torch.float64), dim=0)
    args.output.mkdir(parents=True)
    save_file({'nonclear_mean':small_mean, 'disclosure_mean':large_mean,
               'mean_difference_nonclear_minus_disclosure':difference,
               'nonclear_individuals':small, 'disclosure_individuals':large},
              str(args.output/'group-vectors.safetensors'))
    save_file({'l2_distance':null}, str(args.output/'permutation-null.safetensors'))
    rows=[]
    for layer in range(len(l2)):
        rows.append(dict(layer=layer, cosine_similarity=cosine[layer].item(), l2_distance=l2[layer].item(),
                         null_median=quantiles[0,layer].item(), null_q95=quantiles[1,layer].item(),
                         null_q99=quantiles[2,layer].item(), p_exact_l2=p[layer].item(),
                         p_holm_32_layers=corrected[layer].item()))
    with (args.output/'group-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,9),sharex=True)
    x=list(range(len(l2)))
    axes[0].fill_between(x,quantiles[0],quantiles[1],color='#bdd4e7',alpha=.65,label='Null median to 95th percentile')
    axes[0].plot(x,quantiles[0],color='#255c86',label='Null median')
    axes[0].plot(x,l2,color='#c54e29',label='Observed 4-vs-24')
    axes[0].set_ylabel('L2 between group means'); axes[0].legend(fontsize=8)
    axes[1].plot(x,cosine,color='#255c86'); axes[1].set_ylabel('Cosine similarity of means')
    axes[2].plot(x,p,label='Exact per-layer p')
    axes[2].plot(x,corrected,label='Holm-adjusted p (32 layers)')
    axes[2].axhline(.05,color='gray',linestyle='--',label='0.05')
    axes[2].set_yscale('log'); axes[2].set_ylim(1/len(null)/2,1.2)
    axes[2].set_ylabel('Permutation p-value'); axes[2].set_xlabel('Decoder layer (zero-based)')
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Report-boundary residuals: 4 non-clear disclosures vs 24 disclosures\n'
                 f'Exact label permutation: {len(null):,} unique splits')
    fig.tight_layout(); fig.savefig(args.output/'group-separation.png',dpi=180); plt.close(fig)
    paths=[args.reference/'comparison.safetensors',args.reference/'manifest.json',args.additional/'manifest.json']
    paths += [args.additional/f'trajectory-{i:02d}-boundary.safetensors' for i in NONCLEAR[:-1]]
    manifest=dict(nonclear_indices=NONCLEAR,disclosure_indices=source['disclosure_indices'],
                  model=source['model'],revision=source['revision'],layers=source['layers'],
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  arithmetic='float64; saved boundary vectors only; no model execution in comparison',
                  mean_difference='nonclear mean minus disclosure mean',assignments=len(null),
                  statistic='L2 distance between group means; fixed 4/24 group sizes',
                  p_definition='Exact fraction of all assignments with L2 >= observed, inclusive of observed split; tolerance 1e-10 * max(1, observed L2)',
                  correction='Holm across 32 layerwise L2 tests; cosine similarity descriptive only',
                  caveats='Observational, post-hoc cohort with heterogeneous non-clear reports; not four confirmed deceptions. '
                  'Permutation inference assumes exchangeable labels under the null. Prefix/context differences can explain separation. '
                  'This is exploratory, not causal evidence of a deception feature.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Four-vs-24 report-boundary comparison','',
           'Exact label permutations using L2 between group means; higher separation is more extreme. Cosine similarity is descriptive.','',
           '| Layer | Cosine | L2 | Exact p | Holm p (32 layers) |','| --- | --- | --- | --- | --- |']
    lines += [f"| {r['layer']} | {r['cosine_similarity']:.5f} | {r['l2_distance']:.3f} | {r['p_exact_l2']:.5f} | {r['p_holm_32_layers']:.5f} |" for r in rows]
    lines+=['',manifest['caveats']]
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'permutations':len(null),'min_raw_p':p.min().item(),
                      'min_holm_p':corrected.min().item(),
                      'raw_p_below_05':(p<.05).nonzero().flatten().tolist(),
                      'holm_p_below_05':(corrected<.05).nonzero().flatten().tolist(),
                      'last_layer':rows[-1]},indent=2))


if __name__=='__main__':
    main()
