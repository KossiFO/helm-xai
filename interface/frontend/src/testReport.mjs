// Liste explicite des informations exportées : jamais le texte analysé, les logs,
// les métadonnées d'un modèle ou l'historique complet.
export const ISSUE_URL = "https://github.com/KossiFO/helm-xai/issues/new/choose";
export const MAIL_URL = "mailto:follyametepkossi@gmail.com?subject=Retour%20de%20test%20HELM";

export function buildTestReport(environment = {}, result = null, details = "") {
  const lines = [
    "Retour de test HELM (logiciel, hors étude utilisateur)",
    `Version HELM : ${environment.package_version || "non disponible"}`,
    `Python : ${environment.python || "non disponible"}`,
    `Système : ${environment.system || "non disponible"}`,
  ];
  for (const name of ["numpy", "scipy", "scikit-learn", "shap", "lime", "fastapi"]) {
    if (environment.dependencies?.[name]) lines.push(`${name} : ${environment.dependencies[name]}`);
  }
  if (result) {
    lines.push("", "Dernière analyse affichée (texte non inclus)",
      `Version de l’analyse : ${result.package_version || "non disponible"}`,
      `Modèle : ${result.model}`, `Profil : ${result.profile}`);
    for (const method of Object.keys(result.attributions || {})) {
      lines.push(`Méthode : ${method} ; note locale : ${result.feedback?.[method] ?? "non notée"}`);
    }
  }
  lines.push("", "Mon retour", details.trim() || "[À compléter : étapes, résultat attendu, résultat observé, suggestion.]",
    "", "Les notes ci-dessus sont des avis sur le logiciel, pas des résultats d’étude scientifique.",
    "Ce rapport n’inclut pas automatiquement les commentaires analysés, les logs ou les chemins locaux.");
  return lines.join("\n");
}
