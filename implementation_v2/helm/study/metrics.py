"""Mesures descriptives ; ne remplace pas une analyse par participant/message."""
import math


def _paired_bools(first, second):
    a, b = list(first), list(second)
    if not a or len(a) != len(b) or any(type(x) is not bool for x in a+b):
        raise ValueError("Deux listes non vides de booléens, de même longueur, requises.")
    return a, b


def balanced_control_accuracy(model_correct, participant_accepts):
    """Moyenne acceptations justes / rejets justes ; refuse un seul type de cas."""
    correct, accepts = _paired_bools(model_correct, participant_accepts)
    n_correct = sum(correct)
    n_error = len(correct)-n_correct
    if not n_correct or not n_error:
        raise ValueError("Il faut des décisions correctes ET erronées.")
    accept_rate = sum(c and a for c, a in zip(correct, accepts))/n_correct
    reject_rate = sum(not c and not a for c, a in zip(correct, accepts))/n_error
    return (accept_rate+reject_rate)/2


def confidence_brier(response_correct, confidence):
    """Brier de la probabilité subjective d'avoir répondu juste (0–1).

    Score probabiliste global (plus petit = mieux), pas une mesure de calibration
    pure : compléter par une courbe confiance/exactitude et ses incertitudes.
    """
    correct, conf = list(response_correct), list(confidence)
    if not correct or len(correct) != len(conf) or any(type(v) is not bool for v in correct):
        raise ValueError("Exactitudes booléennes et confiances de même longueur requises.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or not 0 <= v <= 1 for v in conf):
        raise ValueError("Confiances finies dans [0, 1] requises.")
    return sum((p-int(y))**2 for p, y in zip(conf, correct))/len(correct)
