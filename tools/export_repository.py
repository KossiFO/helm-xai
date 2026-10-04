"""Export complete release sources only, without thesis history or research data."""
import argparse
from pathlib import Path
import shutil
from release_sources import source_paths, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Le dossier de destination doit être neuf.')
    paths = source_paths(ROOT)
    output.mkdir(parents=True)
    for path in paths:
        target = output/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (output/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.venv/\n*.egg-info/\nbuild/\ndist/\nnode_modules/\n*.sqlite*\n.env*\n.DS_Store\n')
    write_manifest(output)
    print(f'{len(paths)} sources exportées, sans historique ni données.')


if __name__ == '__main__':
    main()
