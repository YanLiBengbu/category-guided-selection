"""Download the release assets listed in WEIGHTS.json and check each SHA-256."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fetch(url, path, expected):
    if path.exists() and digest(path) == expected:
        print('verified', path.relative_to(ROOT))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'category-guided-selection-download'})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open('wb') as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        if digest(part) != expected:
            raise ValueError('SHA-256 mismatch: ' + path.name)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    print('downloaded', path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='YanLiBengbu/category-guided-selection', help='GitHub OWNER/REPOSITORY')
    parser.add_argument('--frontend-only', action='store_true')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'WEIGHTS.json').read_text(encoding='utf-8'))
    for asset in spec['assets']:
        if args.frontend_only and not asset['destination'].startswith('frontend/'):
            continue
        url = f"https://github.com/{args.repo}/releases/download/{spec['release_tag']}/{asset['name']}"
        fetch(url, ROOT / asset['destination'], asset['sha256'])


if __name__ == '__main__':
    main()
