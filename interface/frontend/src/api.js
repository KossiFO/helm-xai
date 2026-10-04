export async function api(path, body, signal) {
  let response;
  try {
    response = await fetch(`/api${path}`, {
      signal,
      ...(body === undefined
        ? {}
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error(
      "Le service local est arrêté ou inaccessible. Relancez HELM Codex, puis réessayez.",
    );
  }
  let data;
  try {
    data = await response.json();
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new Error("Le service local est momentanément indisponible.");
  }
  if (!response.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : "Vérifiez le texte saisi et les paramètres de votre analyse.";
    throw new Error(detail);
  }
  return data;
}
