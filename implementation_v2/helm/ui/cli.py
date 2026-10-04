"""Lancement local de l’interface distribuée dans la wheel."""
import argparse
from pathlib import Path
import threading
import webbrowser

from helm import __version__


def port_number(value):
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("Le port doit être compris entre 1 et 65535.")
    return port


def main(argv=None):
    parser = argparse.ArgumentParser(description="HELM : explications adaptées au profil, sur cet ordinateur.")
    parser.add_argument("--version", action="version", version=f"helm-xai {__version__}")
    parser.add_argument("--port", type=port_number, default=8767)
    parser.add_argument("--data-dir", type=Path, help="Dossier local pour l’historique et les notes")
    parser.add_argument("--no-browser", action="store_true", help="Ne pas ouvrir le navigateur automatiquement")
    args = parser.parse_args(argv)
    try:
        import uvicorn
        from .app import create_app
    except ImportError as exc:
        parser.exit(1, f"Installez les dépendances de l’interface : pip install 'helm-xai[ui]' (ou la wheel avec [ui]). Détail : {exc}\n")
    database = args.data_dir.expanduser() / "helm.sqlite3" if args.data_dir else None
    app = create_app(database)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"HELM {__version__} — {url}\nArrêt : Ctrl+C", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port))
    done = threading.Event()

    def open_when_ready():
        while not done.wait(0.1):
            if server.started:
                webbrowser.open(url)
                return

    if not args.no_browser:
        threading.Thread(target=open_when_ready, daemon=True).start()
    try:
        server.run()
    finally:
        done.set()
    return 0
