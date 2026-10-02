// Thin wrappers around the HTTP API. Errors carry the server's `detail` message.

async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("The server could not be reached.");
  }
  if (response.ok) return response;
  let detail = `Request failed (${response.status}).`;
  try {
    const body = await response.json();
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // not JSON: keep the generic message
  }
  throw new Error(detail);
}

function postJSON(url, body) {
  return request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((response) => response.json());
}

/** @returns {Promise<object[]>} sample models, then this session's uploads */
export async function listModels() {
  return (await request("/models")).json();
}

/** @returns {Promise<ArrayBuffer>} binary STL whose triangle i is face i of the tools */
export async function fetchMesh(modelId) {
  return (await request(`/models/${encodeURIComponent(modelId)}/mesh`)).arrayBuffer();
}

/** @param {File} file an .stl or .obj file */
export async function uploadModel(file) {
  const form = new FormData();
  form.append("file", file);
  return (await request("/models", { method: "POST", body: form })).json();
}

/** @returns {Promise<{answer: string, trace: object[], view_actions: object[]}>} */
export function ask(question, selection, history) {
  return postJSON("/ask", { question, selection, history });
}
