import { useEffect, useState } from "react";
import { api } from "../api.js";
import { methodName } from "../lib.mjs";

export default function FeedbackPanel({ result, selected, onFeedback }) {
  const [ratings, setRatings] = useState(result.feedback || {});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { setRatings(result.feedback || {}); setError(""); }, [result.feedback, selected]);
  async function rate(rating) {
    setSaving(true); setError("");
    try {
      await api(`/explanations/${result.id}/feedback`, { method: selected, rating });
      setRatings((v) => ({ ...v, [selected]: rating }));
      onFeedback(result.id, selected, rating);
    } catch (e) { setError(e.message); }
    finally { setSaving(false); }
  }
  return (
    <div className="feedback">
      <div><strong>Cette explication vous est-elle utile ?</strong>
        <p>Votre avis sur l’explication affichée · {methodName(selected, result.attributions[selected])}</p></div>
      <div className="rating-buttons" role="group" aria-label="Noter l’explication de 1 à 5">
        {[1, 2, 3, 4, 5].map((n) => <button key={n} disabled={saving} onClick={() => rate(n)}
          aria-label={`Note ${n} sur 5`} aria-pressed={ratings[selected] === n}
          className={ratings[selected] === n ? "rated" : ""}>{n}</button>)}
      </div>
      <p className="feedback-message" aria-live="polite">{error ? <span role="alert">{error}</span>
        : saving ? "Enregistrement…" : ratings[selected] ? "Note enregistrée pour cette explication et ce profil."
        : "1 · Peu utile — 5 · Très utile"}</p>
    </div>
  );
}
