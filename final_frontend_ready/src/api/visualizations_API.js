// src/api/visualization_api.js

const BASE_URL =
  process.env.REACT_APP_API_BASE ||
  (typeof import.meta !== "undefined" ? import.meta.env?.VITE_API_BASE : undefined) ||
  "http://192.168.0.109:8000";

/**
 * Backward-compatible helper that returns ONLY the items array.
 * You can now pass either a chatId string or an options object.
 *
 * @param {string|object} chatOrOpts - chatId string OR options object
 *    When string: treated as chatId.
 *    When object: { chatId, q, limit, offset, order, signal }
 * @param {object} [maybeOpts] - optional options when first arg is a string
 * @returns {Promise<Array>} items
 */
export async function fetchVisualizations(chatOrOpts = "", maybeOpts = {}) {
  const opts =
    typeof chatOrOpts === "string"
      ? { chatId: chatOrOpts, ...maybeOpts }
      : { ...chatOrOpts };

  const { chatId = "", q = "", limit = 100, offset = 0, order = "desc", signal } = opts;

  const params = new URLSearchParams();
  if (chatId) params.set("chat_id", chatId);
  if (q) params.set("q", q);
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  params.set("order", order === "asc" ? "asc" : "desc");

  const url = `${BASE_URL}/api/visualizations/list?${params.toString()}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const msg = await safeErrorText(res, "Failed to load visualizations");
    throw new Error(msg);
  }
  const data = await res.json();
  return data.items || [];
}

/**
 * New: full listing helper. Returns { items, total, chat_ids }
 */
export async function listVisualizations({
  chatId = "",
  q = "",
  limit = 100,
  offset = 0,
  order = "desc",
  signal,
} = {}) {
  const params = new URLSearchParams();
  if (chatId) params.set("chat_id", chatId);
  if (q) params.set("q", q);
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  params.set("order", order === "asc" ? "asc" : "desc");

  const url = `${BASE_URL}/api/visualizations/list?${params.toString()}`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const msg = await safeErrorText(res, "Failed to load visualizations");
    throw new Error(msg);
  }
  return res.json();
}

/**
 * Generate a visualization from a single file upload or a server-side file_path.
 * Provide either `file` (File/Blob) OR `filePath` (string).
 */
export async function generateVisualization({
  question,
  title,
  chatId,
  file,
  filePath,
  signal,
} = {}) {
  if (!question) throw new Error("question is required");
  if (!file && !filePath) throw new Error("Either file or filePath is required");

  const form = new FormData();
  form.append("question", String(question));
  if (title) form.append("title", String(title));
  if (chatId) form.append("chat_id", String(chatId));
  if (file) {
    form.append("file", file);
  } else if (filePath) {
    form.append("file_path", String(filePath));
  }

  const res = await fetch(`${BASE_URL}/api/visualizations/generate`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!res.ok) {
    const msg = await safeErrorText(res, "Failed to generate visualization");
    throw new Error(msg);
  }
  return res.json();
}

/**
 * Generate a visualization by combining multiple server-side files.
 * filePaths can be absolute, or relative to the chat’s Excel folder on the backend.
 */
export async function generateCombinedVisualization({
  filePaths,
  question,
  title,
  chatId,
  signal,
} = {}) {
  if (!Array.isArray(filePaths) || filePaths.length < 2) {
    throw new Error("Provide at least two filePaths");
  }
  if (!question) throw new Error("question is required");

  // Backend accepts form data; file_paths may be JSON string or comma-separated.
  const form = new FormData();
  form.append("question", String(question));
  if (title) form.append("title", String(title));
  if (chatId) form.append("chat_id", String(chatId));
  form.append("file_paths", JSON.stringify(filePaths));

  const res = await fetch(`${BASE_URL}/api/visualizations/generate-combined`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!res.ok) {
    const msg = await safeErrorText(res, "Failed to generate combined visualization");
    throw new Error(msg);
  }
  return res.json();
}

/**
 * URL helpers for rendering images directly (you usually get these in the metadata already).
 */
export const getImageUrl = (plotId) => `${BASE_URL}/api/visualizations/${encodeURIComponent(plotId)}/image`;
export const getThumbUrl = (plotId) => `${BASE_URL}/api/visualizations/${encodeURIComponent(plotId)}/thumb`;

/**
 * Convenience: fetch just the distinct chat_ids available (for filter dropdowns).
 */
export async function fetchVisualizationChatIds({ signal } = {}) {
  const res = await fetch(`${BASE_URL}/api/visualizations/list?limit=1`, { signal });
  if (!res.ok) {
    const msg = await safeErrorText(res, "Failed to load chat ids");
    throw new Error(msg);
  }
  const data = await res.json();
  return data.chat_ids || [];
}

/** Internal: safe error text extraction */
async function safeErrorText(res, fallback) {
  try {
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const j = await res.json();
      return j?.detail || j?.error || fallback;
    }
    return (await res.text()) || fallback;
  } catch {
    return fallback;
  }
}

export { BASE_URL };
