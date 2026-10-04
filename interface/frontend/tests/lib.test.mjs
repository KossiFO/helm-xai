import test from "node:test";
import assert from "node:assert/strict";
import { tokensWithScores, methodMode, methodName } from "../src/lib.mjs";
test("Le surlignage conserve exactement accents, ponctuation et répétitions", () => {
  const text = "Merci ! L’idiot répète : idiot.";
  const tokens = tokensWithScores(text, { Merci: -0.3, idiot: 0.7 });
  assert.equal(tokens.map((t) => t.token).join(""), text);
  assert.equal(
    tokens.filter((t) => t.token === "idiot" && t.score === 0.7).length,
    2,
  );
  assert.equal(tokens[0].score, -0.3);
});
test("La méthode réellement exécutée est identifiée pour les méthodes sans accès au modèle", () => {
  assert.match(
    methodMode("integrated_gradients", { metadata: { mode: "leave_one_out" } }),
    /remplacement/,
  );
  assert.match(
    methodMode("logit_lens", { metadata: { mode: "perturbation_fallback" } }),
    /perturbation/,
  );
  assert.match(
    methodMode("chain_of_thought", { metadata: { mode: "perturbation" } }),
    /perturbation/,
  );
});

test("Un terme conservant la ponctuation reste surligné sans changer le texte", () => {
  const text = "Cet idiot, quel crétin !";
  const tokens = tokensWithScores(text, { "idiot,": 0.4, crétin: 0.6 });
  assert.equal(tokens.map((t) => t.token).join(""), text);
  assert.equal(tokens.find((t) => t.token === "idiot").score, 0.4);
});

test("Les termes entiers avec apostrophe et les sous-termes LIME sont reconnus", () => {
  for (const apostrophe of ["’", "'"]) {
    const text = `C${apostrophe}est un idiot, l${apostrophe}idiot répète une phrase.`;
    const tokens = tokensWithScores(text, {
      [`C${apostrophe}est`]: -0.3,
      idiot: 0.4,
      un: 0.1,
    });
    assert.equal(tokens.map((t) => t.token).join(""), text);
    assert.equal(tokens[0].score, -0.3);
    assert.equal(
      tokens.filter((t) => t.token === "idiot" && t.score === 0.4).length,
      2,
    );
    assert.equal(tokens.find((t) => t.token === "une").score, 0);
  }
});


test("Les anciens résultats IG/CoT/LL ne sont plus nommés comme des méthodes natives", () => {
  assert.equal(methodName("integrated_gradients", { metadata: { mode: "leave_one_out" } }), "LOO (remplacement par le)");
  assert.equal(methodName("logit_lens", { metadata: { mode: "logit_lens" } }), "Variation de norme des états cachés");
  assert.equal(methodName("integrated_gradients", { metadata: { display_name: "IG natif vérifié" } }), "IG natif vérifié");
});
