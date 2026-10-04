const examples = [
  ["Remerciement", "Merci pour cet article intéressant et pour cette réponse utile."],
  ["Désaccord", "Je ne partage pas votre avis, mais je respecte votre point de vue."],
  ["Insulte", "Tu es un idiot stupide, ferme ta gueule."],
];
export default function Controls({ text, setText, model, setModel, models, loading, onAnalyze }) {
  return (
    <section className="card input-card" aria-labelledby="message-title">
      <h2 id="message-title">2. Quel message ?</h2>
      <label className="sr-only" htmlFor="message">Texte à analyser</label>
      <textarea id="message" rows={3} maxLength={2000} value={text}
        onChange={(e) => setText(e.target.value)} placeholder="Collez un commentaire ou écrivez un message…" />
      <div className="input-caption">
        <div className="examples" aria-label="Exemples de messages">
          <span>Exemples :</span>
          {examples.map(([label, sample]) => <button key={label} onClick={() => setText(sample)}>{label}</button>)}
        </div>
        <span className="character-count">{text.length} / 2 000</span>
      </div>
      <div className="input-actions">
        <label htmlFor="model">Modèle</label>
        <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => <option key={m.id} value={m.id} disabled={!m.available}>
            {m.label}{!m.available ? " · non disponible" : ""}
          </option>)}
        </select>
        <button className="primary" onClick={onAnalyze} disabled={loading || !text.trim()}>
          {loading ? "Calcul en cours…" : "Expliquer"}
        </button>
      </div>
      <p className="model-note">{model === "demo"
        ? "Démonstration : modèle pédagogique entraîné sur 12 phrases fabriquées."
        : "Modèle chargé depuis le cache local. Le premier calcul peut prendre plusieurs minutes."}</p>
    </section>
  );
}
