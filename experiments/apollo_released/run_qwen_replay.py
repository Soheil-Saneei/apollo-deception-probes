"""Frozen Qwen replay of the exact released Llama insider-trading transcript text."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

MODEL='Qwen/Qwen3.5-9B'
REVISION='c202236235762e1c871ad0ccb60c8ee5ba337b9a'


def exact_span(tokenizer, ids, offsets, text, start, stop):
    positions=[i for i,(a,b) in enumerate(offsets) if b>start and a<stop]
    if not positions or offsets[positions[0]][0]!=start or offsets[positions[-1]][1]!=stop:
        raise ValueError(f'Cannot align exact character span {start}:{stop}')
    a,b=positions[0],positions[-1]+1
    assert tokenizer.decode(ids[a:b],skip_special_tokens=False,clean_up_tokenization_spaces=False)==text[start:stop]
    return [a,b]


def prepare(tokenizer, source, evaluation, early):
    rows=[]
    for original,row in zip(source,evaluation):
        assert original['trajectory_id']==row['trajectory_id']
        body=original['messages'][-1]['content']
        prefix=tokenizer.apply_chat_template(original['messages'][:-1],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        text=prefix+body
        for m in original['messages'][:-1]:
            assert m['content'] in prefix, 'Native template changed an original message'
        enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        ids,offsets=enc['input_ids'],enc['offset_mapping']
        assert tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)==text
        assistant=exact_span(tokenizer,ids,offsets,text,len(prefix),len(text))
        report_start=original['char_spans']['report'][0]-original['char_spans']['assistant'][0]
        report=exact_span(tokenizer,ids,offsets,text,len(prefix)+report_start,len(text))
        window=early.get(row['trajectory_id'])
        early_span=None
        if window:
            a,b=window['window_char_range']
            early_span=exact_span(tokenizer,ids,offsets,text,len(prefix)+a,len(prefix)+b)
            assert text[len(prefix)+a:len(prefix)+b]==window['baseline_text']
        rows.append(dict(row,messages=original['messages'],llama_input_ids=original['input_ids'],
            qwen_input_ids=ids,qwen_rendered_prefix=prefix,qwen_spans=dict(assistant=assistant,report=report,early=early_span),
            qwen_report_token_count=report[1]-report[0],qwen_pre_report_token_count=report[0],
            qwen_early_token_count=early_span[1]-early_span[0] if early_span else None,
            exact_final_assistant_sha256=hashlib.sha256(body.encode()).hexdigest(),
            activation_path='activations/'+row['trajectory_id']+'.safetensors'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('llama_run',type=Path)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--early-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',default='.hf-cache')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--prepared',action='store_true')
    args=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForImageTextToText
    from safetensors.torch import save_file
    cache=args.cache_dir
    if not args.prepared:
        source=[json.loads(s) for s in (args.llama_run/'manifest.jsonl').read_text().splitlines()]
        evaluation=[json.loads(s) for s in (args.evaluation/'evaluation_manifest.jsonl').read_text().splitlines()]
        early={r['trajectory_id']:r for r in (json.loads(s) for s in (args.early_run/'probe/manifest.jsonl').read_text().splitlines())}
        tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,cache_dir=cache,local_files_only=True)
        rows=prepare(tokenizer,source,evaluation,early)
        assert len(rows)==1000 and len(early)==999
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'activations').mkdir()
        config=dict(model=MODEL,revision=REVISION,dtype='float16',frozen=True,attention='eager',use_cache=False,
            tokenizer=MODEL,tokenizer_revision=REVISION,native_template=True,enable_thinking=False,
            original_generation_tokens_available=False,
            replay='exact released message/completion text, retokenized for Qwen; no generation',
            residual='unnormalized output of all 32 decoder blocks, final assistant tokens',
            length_matching='Llama-defined subset retained; Qwen token counts reported separately',
            source_manifest_sha256=hashlib.sha256((args.llama_run/'manifest.jsonl').read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            examples=1000,early_examples=999,counts=dict(Counter(r['split']+'/'+r['label'] for r in rows)),
            max_sequence_tokens=max(len(r['qwen_input_ids']) for r in rows),
            packages={k:importlib.metadata.version(k) for k in ('torch','transformers','safetensors','tokenizers','flash-linear-attention')})
        (args.output/'config.json').write_text(json.dumps(config,indent=2))
        (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(json.dumps(config),flush=True)
        if args.prepare_only:
            return
    else:
        rows=[json.loads(s) for s in (args.output/'manifest.jsonl').read_text().splitlines()]
        assert not list((args.output/'activations').glob('*.safetensors'))
    model=AutoModelForImageTextToText.from_pretrained(MODEL,revision=REVISION,cache_dir=cache,
        local_files_only=True,dtype=torch.float16,device_map='cuda',attn_implementation='eager').eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    layers=model.model.language_model.layers
    assert len(layers)==32
    captured={}; start=stop=0
    def hook(i):
        def capture(module,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            captured[i]=h[0,start:stop].detach().cpu().contiguous()
        return capture
    handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(layers)]
    def forward(ids):
        inp=torch.tensor([ids],device='cuda')
        return model(input_ids=inp,attention_mask=torch.ones_like(inp),use_cache=False,logits_to_keep=1).logits.detach().cpu()
    def stack():
        values=torch.stack([captured[j] for j in range(32)])
        assert torch.isfinite(values).all()
        return values
    means={k:[] for k in ('report_mean','report_boundary','early_mean')}
    pilot=[]; started=time.monotonic()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            spans=row['qwen_spans']; first,last=spans['assistant']
            start,stop=first-1,last
            logits=forward(row['qwen_input_ids']); values=stack()
            assert torch.isfinite(logits).all()
            if i<4:
                repeated=forward(row['qwen_input_ids'])
                assert torch.equal(values,stack()) and torch.equal(logits,repeated)
                if i==0:
                    for h in handles: h.remove()
                    reference=forward(row['qwen_input_ids'])
                    assert torch.equal(reference,logits)
                    handles=[layer.register_forward_hook(hook(j)) for j,layer in enumerate(layers)]
                errors={}
                for name,end in [('report_boundary',spans['report'][0]),('early',spans['early'][1] if spans['early'] else None)]:
                    if end is None: continue
                    start,stop=end-1,end
                    forward(row['qwen_input_ids'][:end])
                    prefix=stack()[:,0].float()
                    target=values[:,end-first].float()
                    rel=(prefix-target).norm(dim=-1)/target.norm(dim=-1).clamp_min(1e-8)
                    errors[name]=float(rel.max())
                    if rel.max()>.005:
                        raise ValueError(f'Prefix fidelity exceeds 0.5%: {name} {float(rel.max())}')
                pilot.append(dict(trajectory_id=row['trajectory_id'],repeat_equal=True,prefix_relative_l2_max=errors))
                start,stop=first-1,last
            assistant=values[:,1:]
            a,b=spans['report']
            report_mean=assistant[:,a-first:b-first].float().mean(1)
            boundary=values[:,a-first].float()
            if spans['early']:
                a,b=spans['early']
                early_mean=assistant[:,a-first:b-first].float().mean(1)
            else:
                early_mean=torch.zeros_like(report_mean)
            data=dict(resid_post=assistant.contiguous(),assistant_boundary=values[:,0].contiguous(),
                report_mean=report_mean,report_boundary=boundary,early_mean=early_mean)
            save_file(data,str(args.output/row['activation_path']),metadata=dict(trajectory_id=row['trajectory_id'],label=row['label'],split=row['split']))
            for k in means: means[k].append(data[k])
            if i==3:
                (args.output/'pilot_fidelity.json').write_text(json.dumps(dict(passed=True,checks=pilot,hooks_changed_logits=False),indent=2))
                print('PILOT PASSED; scaling to all trajectories',flush=True)
            if (i+1)%25==0 or i==3:
                print(json.dumps(dict(completed=i+1,total=len(rows),elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    for h in handles: h.remove()
    save_file({k:torch.stack(v) for k,v in means.items()} | dict(labels=torch.tensor([int(r['label']=='deceptive') for r in rows]),
        raw_indices=torch.tensor([r['raw_index'] for r in rows]),early_valid=torch.tensor([r['qwen_spans']['early'] is not None for r in rows])),str(args.output/'features.safetensors'))
    (args.output/'summary.json').write_text(json.dumps(dict(completed=1000,honest=666,deceptive=334,layers=32,hidden_size=4096,early_completed=999),indent=2))
    print('FULL QWEN EXTRACTION COMPLETE',flush=True)


if __name__=='__main__':
    main()
