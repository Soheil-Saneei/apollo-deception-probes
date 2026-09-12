"""Download and SHA-256-verify one group from a public artifact release ledger."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ledger', type=Path, help='Downloaded upload-ledger.json')
    parser.add_argument('group', help='Exact group key in the ledger')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    group = json.loads(args.ledger.read_text())['groups'][args.group]
    if not group.get('complete') or not group.get('reconstruction_verified'):
        raise SystemExit('Archive group is not verified complete')
    args.output.mkdir(parents=True, exist_ok=True)
    for asset in [*group['parts'], group['manifest']]:
        if Path(asset['name']).name != asset['name']:
            raise ValueError('Unsafe asset filename')
        if not asset['url'].startswith('https://github.com/Soheil-Saneei/apollo-deception-probes/releases/download/'):
            raise ValueError('Unexpected download origin')
        target = args.output / asset['name']
        if target.exists():
            if target.stat().st_size == asset['bytes'] and digest(target) == asset['sha256']:
                print('Already verified:', target.name)
                continue
            raise FileExistsError(f'Existing file differs; move it before retrying: {target}')
        partial = target.with_name(target.name + '.partial')
        with urllib.request.urlopen(asset['url'], timeout=600) as response, partial.open('wb') as out:
            while chunk := response.read(8 * 1024**2):
                out.write(chunk)
        if partial.stat().st_size != asset['bytes'] or digest(partial) != asset['sha256']:
            raise ValueError('Download verification failed: ' + partial.name)
        partial.rename(target)
        print('Verified:', target.name)


if __name__ == '__main__':
    main()
