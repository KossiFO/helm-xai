import { useEffect, useRef, useState } from "react";
import { api } from "./api.js";
import ProfileSelect from "./components/ProfileSelect.jsx";
import Controls from "./components/Controls.jsx";
import ResultView from "./components/ResultView.jsx";
import History from "./components/History.jsx";

export default function App() {
  const [config, setConfig] = useState(null),
    [profile, setProfile] = useState("utilisateur_final");
  const [text, setText] = useState("");
  const [model, setModel] = useState("demo"),
    [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false),
    [error, setError] = useState(""),
    [page, setPage] = useState("explore");
  const active = useRef(null);
  useEffect(() => {
    const c = new AbortController();
    api("/config", undefined, c.signal)
      .then(setConfig)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => {
      c.abort();
      active.current?.abort();
    };
  }, []);
  function invalidate() {
    active.current?.abort();
    active.current = null;
    setResult(null);
    setLoading(false);
    setError("");
  }
  async function analyze(nextProfile = profile) {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const r = await api(
        "/explanations",
        { text, model, profile: nextProfile },
        controller.signal,
      );
      if (active.current === controller) setResult(r);
    } catch (e) {
      if (e.name !== "AbortError" && active.current === controller)
        setError(e.message);
    } finally {
      if (active.current === controller) setLoading(false);
    }
  }
  function selectProfile(next) {
    if (next === profile) return;
    const hadResult = !!result || loading;
    setProfile(next);
    if (hadResult) analyze(next);
  }
  function rememberFeedback(id, method, rating) {
    setResult((current) =>
      current?.id === id
        ? { ...current, feedback: { ...current.feedback, [method]: rating } }
        : current,
    );
  }
  async function openHistory(id) {
    invalidate();
    const controller = new AbortController();
    active.current = controller;
    setLoading(true);
    setPage("explore");
    try {
      const r = await api(`/explanations/${id}`, undefined, controller.signal);
      if (active.current === controller) {
        setText(r.text);
        setProfile(r.profile);
        setModel(r.model);
        setResult(r);
      }
    } catch (e) {
      if (e.name !== "AbortError" && active.current === controller)
        setError(e.message);
    } finally {
      if (active.current === controller) setLoading(false);
    }
  }
  return (
    <div className="app">
      <a className="skip-link" href="#main">Aller au contenu</a>
      <header className="app-header">
        <h1>HELM <span>— l’explication adaptée au profil</span></h1>
        <p>Choisissez un profil, saisissez un message et explorez son explication.</p>
        <nav aria-label="Navigation principale">
          <button aria-current={page === "explore" ? "page" : undefined}
            className={page === "explore" ? "active" : ""} onClick={() => setPage("explore")}>Expliquer un message</button>
          <button aria-current={page === "history" ? "page" : undefined}
            className={page === "history" ? "active" : ""} onClick={() => setPage("history")}>Historique</button>
        </nav>
      </header>
      <main id="main">
        {error && <p className="error-banner" role="alert">{error}</p>}
        {!config ? (
          <p className="card">{error ? "Le service local doit être démarré." : "Connexion à HELM…"}</p>
        ) : page === "history" ? (
          <History profiles={config.profiles} onOpen={openHistory} />
        ) : (
          <>
            <section className="card" aria-labelledby="profile-title">
              <h2 id="profile-title">1. Qui lit l’explication ?</h2>
              <ProfileSelect profiles={config.profiles} value={profile} onChange={selectProfile} />
            </section>
            <Controls text={text} setText={(v) => { invalidate(); setText(v); }}
              model={model} setModel={(v) => { invalidate(); setModel(v); }}
              models={config.models} loading={loading} onAnalyze={() => analyze()} />
            <ResultView key={result?.id || "pending"} onFeedback={rememberFeedback}
              result={result} loading={loading} />
          </>
        )}
      </main>
      <footer className="page-footer">HELM · Édition Codex · Analyses conservées sur ce poste</footer>
    </div>
  );
}
