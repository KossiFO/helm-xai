"""Explicit release-source inventory; never includes data, environments or study exports."""
import hashlib
import json

ROOT_FILES = ('LICENSE', 'pyproject.toml', 'setup.py', 'MANIFEST.in', 'README.md', 'DROITS.md',
              'CITATION.cff', 'THIRD_PARTY_NOTICES.txt', 'implementation_v2/config.py')
TREES = {
    'implementation_v2/helm': {'.py', '.html', '.js', '.css', '.txt'},
    'implementation_v2/tests': {'.py'}, 'implementation_v2/data': {'.py'},
    'interface/frontend': {'.json', '.js', '.jsx', '.mjs', '.css', '.html'},
    'interface/tests': {'.py'}, 'docs': {'.md'}, 'requirements': {'.txt'},
    'tools': {'.py'}, 'examples': {'.py', '.ipynb', '.md'}, '.github/workflows': {'.yml', '.yaml'},
}
EXCLUDED = {'node_modules', 'dist', '__pycache__', '.venv', '.runtime'}


def source_paths(root):
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    paths += list((root / 'implementation_v2').glob('*.md'))
    for tree, suffixes in TREES.items():
        paths += [p for p in (root/tree).rglob('*') if p.is_file() and p.suffix in suffixes
                  and not any(part in EXCLUDED for part in p.relative_to(root).parts)]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Symlink interdit dans les sources de distribution.')
    return sorted(set(paths))


def write_manifest(root):
    manifest = [{'path': p.relative_to(root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in source_paths(root)]
    (root/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest
