import { methodName, methodMode, pct } from "../lib.mjs";

export default function ResultDetails({ result, selected, onSelect }) {
  const attr = result.attributions[selected];
  const metrics = result.evaluation[selected];
  return (
    <details className="technical-details">
      <summary>{result.profile === "regulateur" ? "Consulter les détails et la traçabilité" : "Consulter les détails techniques"}</summary>
      <p><strong>{methodName(selected, attr)}</strong> · {methodMode(selected, attr)}</p>
      <p className="muted">Probabilité de toxicité : {pct(result.prediction.probabilities[1])}.</p>
      <div className="expert-metrics">
        <div><span>Comprehensiveness C · k = {result.parameters.k}</span><strong>{metrics?.fidelity?.toFixed(3) ?? "Non calculée"}</strong></div>
        <div><span>Suffisance · k = {result.parameters.k}</span><strong>{metrics?.sufficiency?.toFixed(3) ?? "Non calculée"}</strong></div>
        <div><span>Temps de la méthode</span><strong>{attr.computation_time.toFixed(2)} s</strong></div>
      </div>
      <p className="chart-note">{!metrics ? selected === "counterfactual" ? "C/S non applicable à la recherche de contrefactuels. Consulter les variantes et les inversions observées." : "Aucune métrique C/S disponible pour cette explication."
        : metrics.metadata?.protocol === "signed_cs_word_positions_v1"
          ? "Écarts signés pour la classe expliquée (voir métadonnées). Les mots sont classés par importance absolue ; un texte vidé est remplacé par « le »."
          : "Mesures historiques par perturbation de la probabilité de toxicité."}</p>
      {(result.profile === "expert_technique" || result.profile === "regulateur") && (
        <div className="comparison">
          <h3>Comparaison des méthodes</h3>
          <div className="table-scroll"><table><thead><tr><th>Méthode</th><th>Comprehensiveness C</th><th>Suffisance</th><th>Temps</th></tr></thead>
            <tbody>{Object.entries(result.attributions).map(([name, a]) => <tr key={name}>
              <td><button onClick={() => onSelect(name)}>{methodName(name, a)}</button><small>{methodMode(name, a)}</small></td>
              <td>{result.evaluation[name]?.fidelity?.toFixed(3) ?? "—"}</td>
              <td>{result.evaluation[name]?.sufficiency?.toFixed(3) ?? "—"}</td><td>{a.computation_time.toFixed(2)} s</td>
            </tr>)}</tbody></table></div>
        </div>
      )}
      <dl className="audit">
        <div><dt>Identifiant</dt><dd>{result.id}</dd></div>
        <div><dt>Modèle / package</dt><dd>{result.model} / {result.package_version}</dd></div>
        <div><dt>Paramètres</dt><dd>{result.parameters.num_features} features · k = {result.parameters.k} · graine {result.parameters.seed}</dd></div>
        <div><dt>Créée le</dt><dd>{new Date(result.created_at).toLocaleString("fr-FR")}</dd></div>
      </dl>
      <details><summary>Journal et sélection des méthodes</summary>
        <ul>{result.selection.map((s) => <li key={s.method_name}>
          <strong>{methodName(s.method_name, result.attributions[s.method_name])}</strong> : {s.rationale}
        </li>)}</ul><pre>{result.logs.join("\n")}</pre>
      </details>
    </details>
  );
}
