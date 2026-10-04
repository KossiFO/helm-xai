import { useEffect, useRef, useState } from "react";
import { methodName, pct } from "../lib.mjs";
import Attributions from "./Attributions.jsx";
import FeedbackPanel from "./FeedbackPanel.jsx";
import ResultDetails from "./ResultDetails.jsx";

const titles = {
  utilisateur_final: "Comprendre ce classement",
  moderateur: "Examiner ce contenu",
  expert_technique: "Analyser les contributions",
  regulateur: "Retracer cette analyse",
};
export default function ResultView({ result, loading, onFeedback }) {
  const [method, setMethod] = useState(() => Object.keys(result?.attributions || {})[0] || "");
  const heading = useRef(null);
  useEffect(() => { if (result && !loading) heading.current?.focus(); }, [result?.id, loading]);
  if (loading) return <section className="card loading-state" role="status">
    <h2>3. Calcul de l’explication…</h2><p>HELM prépare une explication adaptée au profil choisi.</p>
  </section>;
  if (!result) return null;
  const selected = method in result.attributions ? method : Object.keys(result.attributions)[0];
  const attr = result.attributions[selected];
  if (!attr) return <p className="card" role="status">Aucune explication disponible pour cette analyse.</p>;
  const p = result.prediction.probabilities[1];
  // Seul le contrefactuel de la méthode affichée appartient à l'explication notée.
  const cf = selected === "counterfactual" ? attr.metadata?.counterfactuals?.[0] : null;
  return (
    <section className="card result-card" aria-labelledby="result-title">
      <h2 id="result-title" tabIndex={-1} ref={heading}>3. {titles[result.profile]}</h2>
      <p className="prediction-line">
        <strong className={p > 0.5 ? "risk-tone" : "calm-tone"}>
          {result.prediction.label === "toxique" ? "Potentiellement toxique" : "Non toxique"}
        </strong><span>Confiance du modèle : {pct(result.prediction.confidence)}</span>
      </p>
      {result.profile === "moderateur" && <p className="moderation-note">
        {p > 0.7 ? "Signal élevé à examiner." : p > 0.4 ? "Une lecture humaine est utile." : "Faible signal de toxicité."}
        {" "}La décision de modération vous appartient.
      </p>}
      <div className="method-heading">
        <h3>Les mots qui comptent</h3>
        <label><span className="sr-only">Méthode d’explication</span>
          <select aria-label="Méthode d’explication" value={selected} onChange={(e) => setMethod(e.target.value)}>
            {Object.keys(result.attributions).map((name) => <option key={name} value={name}>
              {methodName(name, result.attributions[name])}
            </option>)}
          </select>
        </label>
      </div>
      <Attributions text={result.text} attr={attr} compact={result.profile === "utilisateur_final" || result.profile === "moderateur"} />
      {cf && <div className="counterfactual"><h3>Un changement qui modifie la prédiction</h3>
        <p>« {cf.counterfactual_text} »</p>
        <small>Après modification : {pct(cf.new_prob)} de probabilité de toxicité.</small>
      </div>}
      {result.warnings.length > 0 && <p className="warning" role="status">
        Certains calculs ou métriques ne sont pas disponibles. Les détails figurent dans le journal.
      </p>}
      <FeedbackPanel key={`${result.id}:${selected}`} result={result} selected={selected} onFeedback={onFeedback} />
      <ResultDetails result={result} selected={selected} onSelect={setMethod} />
      <div className="result-footer"><span>{result.duration.toFixed(2)} s · {Object.keys(result.attributions).length} explications</span>
        <a href={`/api/explanations/${result.id}/export`} download>Exporter l’analyse</a>
      </div>
    </section>
  );
}
