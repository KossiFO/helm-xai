"""
Visualisations pour HELM v2.
Fonctions réutilisables dans les notebooks et scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Optional, Tuple

from helm.config import Attribution, UserProfile, ContentType, DECISION_MATRIX


# ── Style global ──
COLORS = {
    "lime": "#2ecc71",
    "shap": "#e74c3c",
    "integrated_gradients": "#3498db",
    "anchors": "#9b59b6",
    "counterfactual": "#f39c12",
    "chain_of_thought": "#1abc9c",
    "logit_lens": "#e91e63",
    "positive": "#e74c3c",
    "negative": "#2ecc71",
    "neutral": "#95a5a6",
}

PROFILE_COLORS = {
    UserProfile.TECHNICAL_EXPERT: "#3498db",
    UserProfile.MODERATOR: "#e67e22",
    UserProfile.END_USER: "#2ecc71",
    UserProfile.REGULATOR: "#9b59b6",
}

PROFILE_LABELS = {
    UserProfile.TECHNICAL_EXPERT: "Expert\ntechnique",
    UserProfile.MODERATOR: "Modérateur",
    UserProfile.END_USER: "Utilisateur\nfinal",
    UserProfile.REGULATOR: "Régulateur",
}


def plot_attributions(
    attribution: Attribution,
    title: Optional[str] = None,
    max_tokens: int = 15,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Barplot horizontal des attributions token-level."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, max_tokens * 0.35)))

    top = attribution.top_tokens[:max_tokens]
    tokens = [t for t, _ in reversed(top)]
    scores = [s for _, s in reversed(top)]
    colors = [COLORS["positive"] if s > 0 else COLORS["negative"] for s in scores]

    ax.barh(tokens, scores, color=colors, edgecolor="white", height=0.7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("Score d'attribution")
    ax.set_title(
        title or f"Attributions — {attribution.method_name.upper()}",
        fontweight="bold",
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return ax


def plot_method_comparison(
    attributions: Dict[str, Attribution],
    max_tokens: int = 10,
    figsize: Optional[Tuple] = None,
) -> plt.Figure:
    """Comparaison côte à côte des attributions de plusieurs méthodes."""
    n = len(attributions)
    if figsize is None:
        figsize = (6 * n, max(4, max_tokens * 0.35))
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (method, attr) in zip(axes, attributions.items()):
        plot_attributions(attr, max_tokens=max_tokens, ax=ax)
    fig.suptitle(
        "Comparaison inter-méthodes des attributions",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_decision_matrix_heatmap(figsize: Tuple = (16, 5)) -> plt.Figure:
    """Heatmap de la matrice de décision HELM."""
    profiles = list(UserProfile)
    profile_names = [PROFILE_LABELS[p].replace("\n", " ") for p in profiles]
    method_names = ["shap", "lime", "integrated_gradients", "anchors",
                     "counterfactual", "chain_of_thought", "logit_lens"]
    method_labels = ["SHAP", "LIME", "Integrated\nGradients", "Anchors",
                     "Counterfactual", "Chain-of-\nThought", "Logit\nLens"]

    matrix = np.zeros((len(profiles), len(method_names)))
    for i, profile in enumerate(profiles):
        candidates = DECISION_MATRIX.get((profile, ContentType.TEXT), [])
        for method, weight in candidates:
            if method in method_names:
                j = method_names.index(method)
                matrix[i, j] = weight

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.7)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels(method_labels, fontsize=10)
    ax.set_yticks(range(len(profile_names)))
    ax.set_yticklabels(profile_names, fontsize=11)

    for i in range(len(profiles)):
        for j in range(len(method_names)):
            val = matrix[i, j]
            color = "white" if val > 0.4 else ("black" if val > 0 else "#cccccc")
            text = f"{val:.0%}" if val > 0 else "—"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Poids de priorité", shrink=0.8)
    ax.set_title("Matrice de décision HELM", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    return fig


def plot_fidelity_comparison(
    results_df,
    metric: str = "fidelity",
    figsize: Tuple = (10, 6),
) -> plt.Figure:
    """Barplot groupé des métriques par méthode, avec barres d'erreur."""
    fig, ax = plt.subplots(figsize=figsize)

    methods = results_df["method"].unique()
    means = results_df.groupby("method")[metric].mean()
    stds = results_df.groupby("method")[metric].std()

    colors = [COLORS.get(m, "#95a5a6") for m in methods]
    bars = ax.bar(
        [m.upper() for m in methods],
        [means[m] for m in methods],
        yerr=[stds[m] for m in methods],
        color=colors, edgecolor="white", capsize=5, width=0.5,
    )
    for bar, m in zip(bars, methods):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[m] + 0.005,
                f"{means[m]:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Comparaison — {metric.capitalize()} par méthode",
                 fontsize=13, fontweight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def plot_ucb1_evolution(
    history: List[Tuple[int, str, int]],
    methods: List[str],
    figsize: Tuple = (14, 5),
) -> plt.Figure:
    """Évolution des préférences UCB1 au fil des feedbacks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Courbe d'évolution
    for method in methods:
        feedbacks = [(i, r) for i, m, r in history if m == method]
        if feedbacks:
            _, ratings = zip(*feedbacks)
            cumulative = np.cumsum(ratings) / np.arange(1, len(ratings) + 1)
            ax1.plot(range(1, len(ratings) + 1), cumulative, "o-",
                     label=method.upper(), markersize=3, linewidth=1.5,
                     color=COLORS.get(method, "#95a5a6"))

    ax1.set_xlabel("Nombre d'évaluations")
    ax1.set_ylabel("Note moyenne cumulée")
    ax1.set_title("Évolution des préférences", fontweight="bold")
    ax1.legend()
    ax1.set_ylim(1, 5.5)

    # Distribution des notes
    for i, method in enumerate(methods):
        ratings = [r for _, m, r in history if m == method]
        if ratings:
            ax2.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                     alpha=0.6, label=method.upper(),
                     color=COLORS.get(method, "#95a5a6"))

    ax2.set_xlabel("Note (1-5)")
    ax2.set_ylabel("Fréquence")
    ax2.set_title("Distribution des notes", fontweight="bold")
    ax2.legend()

    for ax in [ax1, ax2]:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


def plot_helm_architecture(figsize: Tuple = (14, 10)) -> plt.Figure:
    """Schéma de l'architecture HELM en 3 couches."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def _box(x, y, w, h, color, title, items):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#2c3e50", linewidth=2, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top",
                fontsize=11, fontweight="bold", color="#2c3e50")
        for i, item in enumerate(items):
            ax.text(x + 0.3, y + h - 0.65 - i * 0.35, f"• {item}",
                    fontsize=9, color="#34495e", va="top")

    def _arrow(y_from, y_to):
        ax.annotate("", xy=(5, y_to), xytext=(5, y_from),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#2c3e50"))

    # Entrée
    ax.text(5, 11.5, "Texte + Profil utilisateur", ha="center", fontsize=13,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
            facecolor="#ecf0f1", edgecolor="#2c3e50", linewidth=2))

    _arrow(11.1, 10.6)
    _box(1.5, 9, 7, 1.6, "#3498db",
         "COUCHE 1 — Caractérisation Contextuelle",
         ["Profil utilisateur (expert, modérateur, end-user, régulateur)",
          "Type de contenu (texte, image, multimodal)",
          "Contraintes opérationnelles (temps, détail, audit)"])

    _arrow(9.0, 7.4)
    _box(1.5, 5.8, 7, 1.6, "#e67e22",
         "COUCHE 2 — Sélection Adaptative",
         ["Matrice de décision (profil × contenu → méthodes)",
          "Ajustement par fidélité mesurée (comprehensiveness)",
          "Préférences apprises par UCB1 (bandit multi-bras)"])

    _arrow(5.8, 4.2)
    _box(1.5, 2.6, 7, 1.6, "#2ecc71",
         "COUCHE 3 — Personnalisation Interface",
         ["Expert → Métriques + graphiques détaillés",
          "Modérateur → Dashboard + action recommandée",
          "End-user → Explication en langage naturel",
          "Régulateur → Rapport d'audit traçable"])

    _arrow(2.6, 1.3)
    ax.text(5, 0.8, "Explication adaptée + Métriques", ha="center", fontsize=13,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2c3e50",
            edgecolor="#2c3e50", linewidth=2))

    # Boucle feedback
    ax.annotate("", xy=(9.2, 6.6), xytext=(9.2, 1.0),
                arrowprops=dict(arrowstyle="->", lw=2, color="#e74c3c",
                linestyle="dashed"))
    ax.text(9.5, 3.8, "Feedback\n(UCB1)", ha="left", fontsize=9,
            fontstyle="italic", color="#e74c3c", rotation=90, va="center")

    ax.set_title("Architecture HELM v2", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig
