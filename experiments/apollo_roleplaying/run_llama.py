"""Frozen Llama replay of Qwen's exact roleplaying examples and split assignments."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

MODEL = 'meta-llama/Llama-3.3-70B-Instruct'
REVISION = '6f6073b423013f6a7d4d9f39144961bfbfbc386b'
DATE = '26 Jul 2024'


def encode(tokenizer, row):
    prefix = tokenizer.apply_chat_template(row['messages'], tokenize=False,
        add_generation_prompt=True, date_string=DATE)
    text = prefix + row['completion']
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    positions = [i for i, (a,b) in enumerate(offsets) if b > len(prefix)]
    if not positions or offsets[positions[0]][0] < len(prefix):
        raise ValueError(f"Completion/template token straddles boundary: {row['id']}")
    if tokenizer.decode(ids[positions[0]:], clean_up_tokenization_spaces=False,
                        skip_special_tokens=False) != row['completion']:
        raise ValueError('Exact completion token roundtrip failed')
    return dict(row, input_ids=ids, completion_positions=positions, rendered_prefix=prefix,
                activation_path=f"activations/{row['id']}.safetensors")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',default='.hf-cache-llama')
    args=parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from safetensors.torch import save_file
    from sanity_checks import validate_rows
    source=[json.loads(x) for x in args.qwen_manifest.read_text().splitlines()]
    validate_rows(source)
    assert len(source)==742
    for a,b in zip(source[::2],source[1::2]):
        assert a['scenario_id']==b['scenario_id'] and a['label']=='honest' and b['label']=='deceptive'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.bos_token
    rows=[encode(tokenizer,{k:r[k] for k in ('id','source_index','scenario_id','split','label',
            'messages','completion','completion_sha256')}) for r in source]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'activations').mkdir()
    config=dict(model=MODEL,revision=REVISION,tokenizer=MODEL,tokenizer_revision=REVISION,
        source_qwen_manifest_sha256=hashlib.sha256(args.qwen_manifest.read_bytes()).hexdigest(),
        examples=742,scenario_pairs=371,split_examples=dict(Counter(r['split'] for r in rows)),
        dtype='bfloat16',attention='eager',frozen=True,use_cache=False,
        template_date=DATE,pad_token_id=tokenizer.pad_token_id,
        padding='right-pad each member to pair maximum length; padding masked and excluded',
        pooling='FP32 mean over exact completion tokens only; no end-of-turn token',
        residual='unnormalized output of each decoder block',layers=80,hidden_size=8192,
        original_generation_token_ids_available=False,
        max_input_tokens=max(len(r['input_ids']) for r in rows),
        packages={k:importlib.metadata.version(k) for k in
                  ('torch','transformers','accelerate','huggingface-hub','safetensors','tokenizers')})
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    required=sum(len(r['completion_positions']) for r in rows)*80*8192*2+20_000_000_000
    assert shutil.disk_usage(args.output).free > required
    print(json.dumps(config),flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,cache_dir=args.cache_dir,
        dtype=torch.bfloat16,device_map='balanced',max_memory={i:'43GiB' for i in range(4)},
        attn_implementation='eager').eval().requires_grad_(False)
    assert all(not p.requires_grad and p.device.type=='cuda' for p in model.parameters())
    layers=model.model.layers
    assert len(layers)==80
    (args.output/'device_map.json').write_text(json.dumps({k:str(v) for k,v in model.hf_device_map.items()},indent=2))
    captured={}
    start=stop=0
    def hook(i):
        def capture(module,inputs,output):
            h=output[0] if isinstance(output,tuple) else output
            captured[i]=h[0,start:stop].detach().cpu().contiguous()
        return capture
    handles=[layer.register_forward_hook(hook(i)) for i,layer in enumerate(layers)]
    means,boundaries=[],[]
    fidelity=[]
    begun=time.monotonic()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            start=row['completion_positions'][0]-1
            stop=len(row['input_ids'])
            n=max(len(rows[i//2*2]['input_ids']),len(rows[i//2*2+1]['input_ids']))
            device=model.get_input_embeddings().weight.device
            ids=torch.tensor([row['input_ids']+[tokenizer.pad_token_id]*(n-stop)],device=device)
            mask=torch.tensor([[1]*stop+[0]*(n-stop)],device=device)
            result=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
            if i==0:
                first=torch.stack([captured[j] for j in range(80)])
                repeat=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
                assert torch.equal(result.logits,repeat.logits)
                assert torch.equal(first,torch.stack([captured[j] for j in range(80)]))
                for h in handles: h.remove()
                reference=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
                assert torch.equal(result.logits,reference.logits)
                fidelity.append(dict(check='repeated_forward_and_hooks_vs_no_hooks',max_abs_error=0.))
                handles=[layer.register_forward_hook(hook(j)) for j,layer in enumerate(layers)]
            values=torch.stack([captured[j] for j in range(80)])
            assert torch.isfinite(values).all()
            boundary,completion=values[:,0],values[:,1:]
            means.append(completion.float().mean(1))
            boundaries.append(boundary.float())
            if i%2:
                error=(boundaries[-1]-boundaries[-2]).abs().max().item()
                if error!=0:
                    raise ValueError(f'Paired prompt boundary mismatch {row["id"]}: {error}')
            path=args.output/row['activation_path']
            temporary=path.with_suffix('.tmp')
            save_file(dict(resid_post=completion.contiguous(),prompt_boundary=boundary.contiguous()),
                str(temporary),metadata=dict(example_id=row['id'],label=row['label'],split=row['split'],
                                            axes='layer,completion_token,hidden'))
            temporary.replace(path)
            if i==3:
                (args.output/'pilot_fidelity.json').write_text(json.dumps(dict(passed=True,
                    examples=4,paired_prompt_max_abs_error=0.,checks=fidelity),indent=2))
                print('PILOT PASSED; continuing all 742 completions',flush=True)
            if i%20==0 or i==741:
                print(json.dumps(dict(completed=i+1,total=742,elapsed_seconds=round(time.monotonic()-begun,1))),flush=True)
    for h in handles: h.remove()
    save_file(dict(mean_resid_post=torch.stack(means),prompt_boundary=torch.stack(boundaries)),
              str(args.output/'features.safetensors'))
    (args.output/'fidelity.json').write_text(json.dumps(dict(passed=True,examples=742,
        paired_prompts_checked=371,paired_prompt_max_abs_error=0.,checks=fidelity),indent=2))
    (args.output/'summary.json').write_text(json.dumps(dict(completed=742,honest=371,deceptive=371,
        feature_shape=[742,80,8192]),indent=2))
    print('FULL EXTRACTION COMPLETE',flush=True)


if __name__=='__main__':
    main()
