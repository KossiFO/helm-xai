"""Construit les fichiers front, la wheel autonome et le sdist, sans publication."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from release_sources import write_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-npm-install", action="store_true")
    parser.add_argument("--no-build-isolation", action="store_true",
                        help="Utiliser les outils de build déjà installés dans l'environnement courant")
    parser.add_argument("--outdir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    out = args.outdir.resolve()
    if out.exists() and any(out.glob(f"helm_xai-{version}*")):
        parser.error(f"La version {version} existe déjà dans {out}. Choisir un autre dossier ou une nouvelle version.")
    frontend = ROOT / "interface/frontend"
    npm = shutil.which("npm")
    if not npm:
        parser.error("Node.js/npm est nécessaire pour construire les sources, pas pour installer la wheel.")
    if not args.skip_npm_install:
        subprocess.run([npm, "ci"], cwd=frontend, check=True)
    subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
    static = ROOT / "implementation_v2/helm/ui/static"
    if static.exists():
        shutil.rmtree(static)  # uniquement les fichiers reconstruits de l’interface
    shutil.copytree(frontend / "dist", static)
    notices = ["HELM : notices des composants JavaScript redistribués\n"]
    for name in ("react", "react-dom", "scheduler", "lucide-react", "vite"):
        module = frontend / "node_modules" / name
        metadata = json.loads((module / "package.json").read_text())
        license_file = next((p for p in (module / "LICENSE", module / "LICENSE.txt", module / "LICENSE.md") if p.is_file()), None)
        if license_file is None:
            raise RuntimeError(f"Licence manquante : {name}")
        notices.append(f"\n{'=' * 72}\n{name} {metadata['version']}\n{'=' * 72}\n" + license_file.read_text())
    notice_text = "\n".join(notices)
    (ROOT / "THIRD_PARTY_NOTICES.txt").write_text(notice_text)
    (static / "THIRD_PARTY_NOTICES.txt").write_text(notice_text)
    write_manifest(ROOT)
    build_command = [sys.executable, "-m", "build", "--outdir", str(out)]
    if args.no_build_isolation:
        build_command.append("--no-isolation")
    subprocess.run(build_command, cwd=ROOT, check=True)
    artifacts = sorted(out.glob(f"helm_xai-{version}*"))
    checksums = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in artifacts if p.is_file())
    (out / f"SHA256SUMS-{version}.txt").write_text(checksums)
    print(checksums)


if __name__ == "__main__":
    main()
