import { buildTestReport, ISSUE_URL, MAIL_URL } from "../testReport.mjs";

export default function TesterSupport({ environment, result, details, setDetails }) {
  const report = buildTestReport(environment, result, details);
  function download() {
    const url = URL.createObjectURL(new Blob([report], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url; link.download = "retour-test-helm.txt";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="card tester-support" aria-labelledby="tester-title">
    <h2 id="tester-title">Tester HELM et faire un retour</h2>
    <p>Premier essai : gardez « Démonstration locale », choisissez un exemple et cliquez sur « Expliquer ».
      Comparez ensuite les profils Utilisateur et Expert technique sur le même message.</p>
    <p>La démonstration utilise 12 phrases fabriquées. Elle sert à essayer le logiciel,
      pas à juger la toxicité réelle d’un commentaire ni à valider l’adaptation aux profils.</p>
    <p><strong>Vos notes restent sur cet ordinateur. Elles ne sont pas envoyées à l’auteur.</strong>
      {" "}Pour partager un retour sur l’installation, un problème ou une explication peu claire :</p>
    <ol>
      <li>Décrivez les étapes, ce que vous attendiez et ce que vous avez observé.</li>
      <li>Relisez le rapport ci-dessous, puis téléchargez-le.</li>
      <li>Copiez son contenu dans une issue GitHub ou joignez le fichier à votre courriel.</li>
    </ol>
    <label htmlFor="tester-details">Votre retour (sans données personnelles ou confidentielles)</label>
    <textarea id="tester-details" rows={4} maxLength={4000} value={details}
      onChange={(e) => setDetails(e.target.value)} placeholder="Étapes suivies · Résultat attendu · Résultat observé · Suggestion" />
    <label htmlFor="tester-report">Rapport à relire avant de le partager</label>
    <textarea id="tester-report" rows={9} readOnly value={report} />
    <div className="support-actions">
      <button onClick={download}>Télécharger mon retour</button>
      <a href={ISSUE_URL} target="_blank" rel="noreferrer">Ouvrir GitHub (compte requis)</a>
      <a href={MAIL_URL}>Préparer un courriel</a>
    </div>
    <p>Les issues GitHub sont publiques. Les liens ci-dessus n’envoient ni votre rapport ni vos notes.
      Le courriel s’ouvre dans votre messagerie configurée ; sinon écrivez à follyametepkossi@gmail.com.
      Le rapport et le texte saisi ici ne sont pas conservés après fermeture de la page.</p>
  </section>;
}
