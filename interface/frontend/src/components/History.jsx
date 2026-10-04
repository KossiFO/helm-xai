import { useEffect, useState } from "react";
import { ArrowUpRight, History as HistoryIcon } from "lucide-react";
import { api } from "../api.js";
import { pct } from "../lib.mjs";
export default function History({ profiles, onOpen }) {
  const [items, setItems] = useState(null),
    [error, setError] = useState("");
  useEffect(() => {
    const c = new AbortController();
    api("/history", undefined, c.signal)
      .then(setItems)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => c.abort();
  }, []);
  return (
    <section className="card history-card">
      <div className="card-heading">
        <HistoryIcon size={20} />
        <h2>Vos analyses sur ce poste</h2>
      </div>
      <p className="muted">
        Retrouvez le texte, le profil et les explications enregistrés au moment
        du calcul.
      </p>
      {error && <p role="alert">{error}</p>}
      {items === null ? (
        <p>Chargement…</p>
      ) : items.length === 0 ? (
        <div className="history-empty">
          Votre première analyse apparaîtra ici.
        </div>
      ) : (
        <div className="history-list">
          {items.map((r) => (
            <button key={r.id} onClick={() => onOpen(r.id)}>
              <div>
                <span className="history-meta">
                  {profiles.find((p) => p.id === r.profile)?.label} · {r.model}{" "}
                  · {new Date(r.created_at).toLocaleString("fr-FR")}
                </span>
                <strong>{r.text}</strong>
              </div>
              <span className="history-score">
                {pct(r.prediction.probabilities[1])}
                <small>toxicité</small>
              </span>
              <ArrowUpRight size={18} />
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
