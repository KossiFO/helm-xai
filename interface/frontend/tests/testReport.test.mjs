import test from "node:test";
import assert from "node:assert/strict";
import { buildTestReport, ISSUE_URL, MAIL_URL } from "../src/testReport.mjs";

test("le rapport partage les versions et les notes, pas les données analysées", () => {
  const environment = { package_version: "0.test", python: "3.13", system: "Darwin",
    dependencies: { numpy: "2.0", secret: "DEPENDENCY_SECRET" }, home: "PRIVATE_HOME" };
  const result = { package_version: "0.old", model: "demo", profile: "moderateur", text: "PRIVATE_TEXT",
    id: "PRIVATE_ID", logs: ["PRIVATE_LOG"], attributions: { lime: { metadata: { secret: "PRIVATE_METADATA" } } },
    feedback: { lime: 4 }, history: ["PRIVATE_HISTORY"] };
  const report = buildTestReport(environment, result, "Le libellé reste difficile à comprendre.");
  assert.match(report, /Version HELM : 0.test/);
  assert.match(report, /Version de l’analyse : 0.old/);
  assert.match(report, /lime ; note locale : 4/);
  assert.match(report, /difficile à comprendre/);
  assert.doesNotMatch(report, /PRIVATE_|DEPENDENCY_SECRET/);
});

test("un retour reste possible avant toute analyse ou si le service ne répond pas", () => {
  assert.match(buildTestReport(), /Version HELM : non disponible/);
  assert.match(buildTestReport(undefined, null, "Installation bloquée"), /Installation bloquée/);
  assert.doesNotMatch(buildTestReport(), /Dernière analyse/);
});

test("les liens ne transmettent pas le contenu du rapport", () => {
  assert.equal(new URL(ISSUE_URL).search, "");
  assert.equal(new URL(MAIL_URL).searchParams.has("body"), false);
});
