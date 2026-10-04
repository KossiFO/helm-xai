"""Recette logicielle synthétique, sans participants ni modèle distant."""
import argparse
import json
from pathlib import Path
import numpy as np
from helm.study import FixedStudyConfig, prepare_stimulus, export_stimulus, participant_payload


def synthetic_predict(texts):
    # Règle pédagogique, pas un modèle validé de toxicité.
    p = np.array([.8 if "idiot" in text.split() else .2 for text in texts])
    return np.column_stack([1-p, p])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index, text in enumerate(["idiot merci merci", "merci pour cette réponse"]):
        for method in ["leave_one_out", "shap"]:
            record = prepare_stimulus(text, synthetic_predict,
                                      FixedStudyConfig(method, "synthetic-keyword-rule-v1"),
                                      case_id=f"synthetic-{index}")
            export_stimulus(record, args.output / f"research-{index}-{method}.json")
            if record["status"] != "ok":
                raise RuntimeError(record["error"])
            # Cette recette n'est pas une plateforme de passation. Les noms de
            # fichiers de recherche ne doivent pas être exposés aux participants.
            with (args.output / f"preview-{index}-{method}.json").open("x") as stream:
                json.dump(participant_payload(record), stream, ensure_ascii=False, indent=2)
            print(f"{index} / {method} : {record['status']} — {record['sha256']}")


if __name__ == "__main__":
    main()
