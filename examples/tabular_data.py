"""Synthetic demonstration only; no insurance data or realistic performance claim."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def prepare_demo():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"montant": rng.uniform(100, 6000, 200),
                      "anciennete": rng.uniform(0, 20, 200),
                      "type_contrat": rng.choice(["auto", "habitation", "autre"], 200),
                      "canal": rng.choice(["agence", "internet"], 200)})
    y = ((X.montant > 3500) & ((X.anciennete < 8) | (X.canal == "internet"))).astype(int)
    train, test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    prep = ColumnTransformer([("quant", "passthrough", ["montant", "anciennete"]),
                              ("qual", OneHotEncoder(handle_unknown="ignore"), ["type_contrat", "canal"])])
    model = make_pipeline(prep, RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42))
    model.fit(train, y_train)
    return model, train, test, y_test
