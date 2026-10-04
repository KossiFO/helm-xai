"""Train-only preparation of numeric inputs and nominal categories."""

import pandas as pd
from sklearn.impute import SimpleImputer

NUMERIC = [
    "age", "delai_sin", "max_diff_capital", "nb_banlieues_a",
    "nb_banlieues_l", "nb_contacts12", "nb_contrats",
    "nb_difficultes_q", "ratio_capital_surface",
]
CATEGORICAL = [
    "code_formule_covea", "code_fractionnement", "code_region",
    "code_structure_foyer", "code_type_risque", "lib_circonstance",
    "qualif_payeur", "rg_categorie_sociop",
]
FEATURES = NUMERIC + CATEGORICAL
MISSING = "__MANQUANTE__"
UNKNOWN = "__INCONNUE__"


def normalize(frame):
    result = frame[FEATURES].copy()
    result[NUMERIC] = result[NUMERIC].apply(pd.to_numeric, errors="raise")
    for name in CATEGORICAL:
        values = result[name].astype("string")
        assert not values.isin([MISSING, UNKNOWN]).any(), "Reserved category collision"
        result[name] = values.fillna(MISSING).astype(object)
    return result


def fit_schema(raw_train):
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(raw_train[NUMERIC])
    categories = {name: sorted(raw_train[name].unique().tolist()) for name in CATEGORICAL}
    return {"imputer": imputer, "categories": categories}


def transform(raw, schema):
    result = raw.copy()
    result[NUMERIC] = schema["imputer"].transform(raw[NUMERIC])
    for name in CATEGORICAL:
        known = schema["categories"][name]
        vocabulary = sorted(set(known) | {MISSING, UNKNOWN})
        values = raw[name].where(raw[name].isin(known + [MISSING]), UNKNOWN)
        result[name] = pd.Categorical(values, categories=vocabulary, ordered=False)
    return result
