import { tokensWithScores, signed } from "../lib.mjs";
export default function Attributions({ text, attr, compact = false }) {
  if (attr.metadata?.score_kind === "norm_growth_proxy" || attr.metadata?.mode === "logit_lens")
    return <p className="chart-note">Ce diagnostic mesure la variation des états internes du modèle.
      Il n’indique pas une contribution vers une classe. Les sous-tokens sont conservés dans l’export.</p>;
  // La couleur reste orientée vers la toxicité, y compris pour une attribution de classe 0.
  const scores = Object.fromEntries(Object.entries(attr.token_scores).map(([word, value]) =>
    [word, attr.metadata?.target_class === 0 ? -value : value]));
  const top = Object.entries(scores)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, compact ? 5 : 8);
  const max = Math.max(...top.map(([, s]) => Math.abs(s)), 0.001);
  return (
    <>
      <div className="highlighted-text" aria-label="Texte et mots importants">
        {tokensWithScores(text, scores).map(({ token, score }, i) => (
          <span
            key={i}
            className={
              score > 0 ? "token toxic" : score < 0 ? "token reassuring" : ""
            }
            title={score ? `${token} : ${signed(score)}` : undefined}
          >
            {token}
          </span>
        ))}
      </div>
      <div className="legend">
        <span>
          <i className="legend-dot risk" />
          Vers « toxique »
        </span>
        <span>
          <i className="legend-dot calm" />
          Vers « non toxique »
        </span>
      </div>
      <details className="word-details">
        <summary>Voir le poids des mots</summary>
      <div className="contribution-list">
        {top.map(([token, score]) => (
          <div className="contribution" key={token}>
            <span className="word">{token}</span>
            <div className="bar-track">
              <div
                className={`bar ${score >= 0 ? "positive" : "negative"}`}
                style={{ width: `${(Math.abs(score) / max) * 48}%` }}
              />
            </div>
            <span
              className={
                score >= 0 ? "score positive-text" : "score negative-text"
              }
            >
              {signed(score)}
            </span>
          </div>
        ))}
      </div>
      {top.length === 0 && (
        <p className="muted">
          Cette méthode n’a pas retenu de mot déterminant pour ce texte.
        </p>
      )}
      <p className="chart-note">
        Importance relative des mots pour cette méthode. Ces scores ne sont pas
        des probabilités.
      </p>
      </details>
    </>
  );
}
