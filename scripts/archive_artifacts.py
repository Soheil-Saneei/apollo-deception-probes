"""One-off VM retirement: inventory research files and upload verified release parts.

Run inventory first. Upload accepts a GitHub token on stdin, never in a file or argv.
Model caches, environments, credentials, and machine configuration are excluded.
Requires GNU tar and zstd on the source machine. No source file is deleted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PART_SIZE = 1024 ** 3
PATTERNS = [rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----",
            rb"\b(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{25,}",
            rb"\bhf_[A-Za-z0-9]{25,}", rb"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{30,}"]
SKIP = {"__pycache__", ".git", ".lavish", ".DS_Store", "node_modules"}


def stream_digest(stream):
    h = hashlib.sha256()
    while chunk := stream.read(8 * 1024 ** 2):
        h.update(chunk)
    return h.hexdigest()


def eligible(p, root):
    rel = p.relative_to(root)
    return not any(x in SKIP or x.startswith((".venv", ".hf-cache", ".env")) for x in rel.parts) and p.suffix not in {".pem", ".key", ".pyc"}


def inventory(root, work, source):
    rows, hits, excluded = [], [], []
    # Do not traverse caches, environments, or hidden machine configuration.
    candidates = []
    for p in root.iterdir():
        if p.name.startswith('.') or not eligible(p, root):
            excluded.append(p.name)
        elif p.is_dir():
            candidates.extend(f for f in p.rglob('*') if f.is_file())
        elif p.is_file(): candidates.append(p)
    for p in sorted(candidates):
        if not eligible(p, root):
            excluded.append(str(p.relative_to(root)))
            continue
        if p.is_symlink():
            raise RuntimeError(f"Review symlink before archiving: {p}")
        h = hashlib.sha256()
        is_text = p.suffix.lower() not in {'.safetensors', '.pt', '.pth', '.npz', '.npy', '.png', '.pdf', '.jpg', '.joblib', '.pkl', '.gz', '.zst'}
        prev = b''
        with p.open('rb') as f:
            while chunk := f.read(8 * 1024 ** 2):
                h.update(chunk)
                if is_text and any(re.search(pattern, prev + chunk) for pattern in PATTERNS):
                    hits.append(str(p.relative_to(root)))
                prev = chunk[-200:]
        rel = str(p.relative_to(root))
        parts = p.relative_to(root).parts
        if 'outputs' in parts and parts.index('outputs') + 1 < len(parts):
            i = parts.index('outputs')
            group = '-'.join(parts[1:i] + (parts[i + 1],)) if parts[0] == 'experiments' else 'legacy-' + parts[i + 1]
        else: group = 'code-and-inputs'
        rows.append(dict(path=rel, bytes=p.stat().st_size, sha256=h.hexdigest(), group=group))
        if len(rows) % 500 == 0: print(f"inventoried {len(rows)} files", flush=True)
    result = dict(source=source, files=rows, excluded=excluded, secret_pattern_hits=sorted(set(hits)))
    (work/'inventory.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(files=len(rows), bytes=sum(r['bytes'] for r in rows), secret_hits=len(hits))), flush=True)
    if hits: raise RuntimeError('Publication blocked: review secret-pattern file paths in inventory')


def api(url, token, data=None, size=None, method=None):
    headers = {'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json', 'User-Agent': 'apollo-research-archive', 'X-GitHub-Api-Version': '2022-11-28'}
    if size is not None: headers.update({'Content-Type':'application/octet-stream', 'Content-Length':str(size)})
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=600) as response:
        if response.status == 204: return None
        return json.load(response)


def upload_file(path, repo, release_id, token):
    with path.open('rb') as stream:
        digest = stream_digest(stream)
    base = f'https://api.github.com/repos/{repo}/releases/{release_id}/assets'
    # Recover safely after an upload succeeded but its response was lost.
    for attempt in range(5):
        try:
            assets = []
            for page in range(1, 12):
                batch = api(base+f'?per_page=100&page={page}', token)
                assets.extend(batch)
                if len(batch) < 100: break
            existing = next((a for a in assets if a['name'] == path.name), None)
            if existing is not None and existing['state'] == 'starter':
                api(existing['url'], token, method='DELETE')
                existing = None
            if existing is None:
                url = f'https://uploads.github.com/repos/{repo}/releases/{release_id}/assets?name='+urllib.parse.quote(path.name)
                with path.open('rb') as stream: existing = api(url, token, stream, path.stat().st_size)
            if existing['size'] != path.stat().st_size or existing.get('digest') != 'sha256:'+digest:
                raise ValueError('Uploaded asset digest/size mismatch: '+path.name)
            return dict(name=path.name, bytes=existing['size'], sha256=digest, url=existing['browser_download_url'], asset_id=existing['id'])
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f'upload retry {attempt+1}: {path.name}: {type(error).__name__}', flush=True)
            if attempt == 4: raise
            time.sleep(10)


def upload(root, work, source, repo, release_id):
    token = sys.stdin.readline().strip()
    if not token: raise RuntimeError('Missing token on stdin')
    inv = json.loads((work/'inventory.json').read_text())
    if inv['secret_pattern_hits']: raise RuntimeError('Unresolved secret scan')
    ledger_path = work/'upload-ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(source=source, repo=repo, release_id=release_id, groups={})
    groups = sorted(set(r['group'] for r in inv['files']))
    for group in groups:
        if ledger['groups'].get(group, {}).get('complete'): continue
        rows = [r for r in inv['files'] if r['group'] == group]
        prefix = source+'--'+group
        listing = work/(prefix+'.files')
        listing.write_bytes(b''.join(r['path'].encode()+b'\0' for r in rows))
        manifest = work/(prefix+'.manifest.json')
        manifest.write_text(json.dumps(dict(source=source, files=rows), indent=2)+'\n')
        parts = []
        archive = subprocess.Popen(['tar','--sort=name','--mtime=2026-09-12','--owner=0','--group=0','--numeric-owner','-C',str(root),'--null','-T',str(listing),'-I','zstd -T4 -3','-cf','-'], stdout=subprocess.PIPE)
        pool = ThreadPoolExecutor(max_workers=4)
        pending = []
        def finish_one():
            record = pending.pop(0).result()
            parts.append(record)
            print(f"verified {record['name']} ({record['bytes']/1024**2:.1f} MiB)", flush=True)
        try:
            index = 0
            while True:
                part = work/(prefix+f'.tar.zst.part{index:04d}')
                with part.open('wb') as out:
                    remaining = PART_SIZE
                    while remaining:
                        chunk = archive.stdout.read(min(8*1024**2, remaining))
                        if not chunk: break
                        out.write(chunk); remaining -= len(chunk)
                if part.stat().st_size == 0:
                    part.unlink(); break
                pending.append(pool.submit(upload_file, part, repo, release_id, token))
                if len(pending) >= 4: finish_one()
                index += 1
            while pending: finish_one()
            if archive.wait() != 0: raise RuntimeError('tar failed: '+group)
        finally:
            pool.shutdown(wait=True)
            if archive.poll() is None: archive.terminate(); archive.wait()
        # Verify the archive stream decodes and every restored byte matches the original manifest.
        import tarfile
        cat = subprocess.Popen(['cat']+[str(work/r['name']) for r in parts], stdout=subprocess.PIPE)
        dec = subprocess.Popen(['zstd','-dc'], stdin=cat.stdout, stdout=subprocess.PIPE)
        cat.stdout.close()
        expected = {r['path']: r for r in rows}
        seen = set()
        with tarfile.open(fileobj=dec.stdout, mode='r|', bufsize=8 * 1024**2) as tf:
            for member in tf:
                if not member.isfile(): raise RuntimeError('Unexpected archive entry')
                record = expected[member.name]
                with tf.extractfile(member) as f: digest = stream_digest(f)
                if digest != record['sha256'] or member.size != record['bytes']: raise RuntimeError('Archive reconstruction mismatch: '+member.name)
                seen.add(member.name)
        # Drain decompressor output so trailer checks complete without SIGPIPE.
        while dec.stdout.read(1024**2): pass
        if dec.wait() or cat.wait() or seen != set(expected): raise RuntimeError('Incomplete archive reconstruction')
        manifest_asset = upload_file(manifest, repo, release_id, token)
        ledger['groups'][group] = dict(complete=True, files=len(rows), uncompressed_bytes=sum(r['bytes'] for r in rows), parts=parts, manifest=manifest_asset, reconstruction_verified=True)
        ledger_path.write_text(json.dumps(ledger, indent=2)+'\n')
        print('COMPLETE '+group, flush=True)
    upload_file(work/'inventory.json', repo, release_id, token)
    upload_file(ledger_path, repo, release_id, token)
    print('ALL_ARTIFACTS_VERIFIED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['inventory','upload'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--repo', default='Soheil-Saneei/apollo-deception-probes')
    p.add_argument('--release-id', type=int)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.mode == 'inventory': inventory(args.root, args.work, args.source)
    else: upload(args.root, args.work, args.source, args.repo, args.release_id)
