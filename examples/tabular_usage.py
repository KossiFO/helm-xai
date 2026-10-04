"""Run from this repository: python examples/tabular_usage.py [output.html]."""
import sys
from pathlib import Path

from helm import TabularHELM
from tabular_data import prepare_demo

model, X_train, X_test, y_test = prepare_demo()
helm = TabularHELM(model, X_train, target_class=1)
report = helm.explain_top(X_test, k=5, method="shap", profile="metier", labels=y_test)

output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("helm_tabulaire_demo.html")
html = report.to_html().replace("<body>", "<body><p><b>Démonstration sur des données entièrement synthétiques — aucun résultat MAAF.</b></p>")
output.write_text(html, encoding="utf-8")
print("Démonstration synthétique —", len(X_train), "lignes d’apprentissage,", len(X_test), "lignes test")
print(report.ranking.to_string())
print("Explications réussies :", len(report.explanations), "/", len(report.ranking))
print("Scores les plus bas :")
print(helm.rank(X_test, k=5, order="lowest").to_string())
print("Rapport :", output.resolve())
if report.errors:
    raise RuntimeError(report.errors)
