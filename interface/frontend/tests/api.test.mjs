import test from "node:test";
import assert from "node:assert/strict";
import { api } from "../src/api.js";

test("Une connexion refusée donne un message de relance compréhensible", async (t) => {
  t.mock.method(globalThis, "fetch", async () => { throw new TypeError("Failed to fetch"); });
  await assert.rejects(api("/config"), /service local.*Relancez HELM Codex/);
});

test("Une requête annulée reste une annulation sans message de panne", async (t) => {
  const aborted = new DOMException("Aborted", "AbortError");
  t.mock.method(globalThis, "fetch", async () => { throw aborted; });
  await assert.rejects(api("/config"), (error) => error === aborted);
});

test("Une annulation pendant la lecture JSON est également conservée", async (t) => {
  const aborted = new DOMException("Aborted", "AbortError");
  t.mock.method(globalThis, "fetch", async () => ({ok: true, json: async () => { throw aborted; }}));
  await assert.rejects(api("/config"), (error) => error === aborted);
});
