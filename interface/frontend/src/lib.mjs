export const METHOD_NAMES = {
  lime: "LIME",
  shap: "SHAP",
  integrated_gradients: "Gradients intégrés",
  anchors: "Anchors",
  counterfactual: "Contrefactuels",
  chain_of_thought: "Chain of Thought",
  logit_lens: "Logit Lens",
};
export const pct = (value) => `${Math.round(value * 100)} %`;
export const signed = (value) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
export function tokensWithScores(text, scores) {
  const normalized = new Map(
    Object.entries(scores)
      .map(([term, score]) => [
        term
          .trim()
          .toLocaleLowerCase("fr")
          .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""),
        score,
      ])
      .filter(([term]) => term.length),
  );
  // Priorité aux termes attribués entiers (apostrophes et expressions incluses).
  // Les frontières empêchent « un » de surligner une partie de « une ».
  const alternatives = [...normalized.keys()]
    .sort((a, b) => b.length - a.length)
    .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const attributed = alternatives.length
    ? `(?<![\\p{L}\\p{N}])(?:${alternatives.join("|")})(?![\\p{L}\\p{N}])|`
    : "";
  const pattern = new RegExp(
    attributed + "[\\p{L}\\p{N}]+|[^\\p{L}\\p{N}]+",
    "giu",
  );
  return (text.match(pattern) || []).map((token) => ({
    token,
    score: normalized.get(token.toLocaleLowerCase("fr")) || 0,
  }));
}
export function methodName(name, attr) {
  if (attr?.metadata?.display_name) return attr.metadata.display_name;
  const mode = attr?.metadata?.mode;
  if (name === "integrated_gradients" && mode === "leave_one_out") return "LOO (remplacement par le)";
  if (name === "chain_of_thought" && mode === "perturbation") return "Perturbations et texte à gabarit";
  if (name === "logit_lens") return mode === "logit_lens" ? "Variation de norme des états cachés" : "Perturbations par suppression";
  if (name === "anchors") return mode === "native" ? "Anchors (bibliothèque officielle)" : "Anchors (recherche interne)";
  return METHOD_NAMES[name] || name;
}
export function methodMode(name, attr) {
  const mode = attr?.metadata?.mode;
  if (name === "integrated_gradients" && mode === "leave_one_out")
    return "LOO · remplacement d’un mot à la fois";
  if (name === "chain_of_thought" && mode === "perturbation")
    return "Attributions par perturbation";
  if (name === "logit_lens" && mode === "perturbation_fallback")
    return "Attributions par perturbation";
  if (name === "anchors")
    return mode === "native"
      ? "Implémentation native"
      : "Recherche interne par faisceau";
  return name === "counterfactual"
    ? "Perturbations du texte"
    : name === "lime"
      ? "Modèle linéaire local"
      : name === "shap"
        ? "Attributions de Shapley"
        : mode || "Calcul HELM";
}
