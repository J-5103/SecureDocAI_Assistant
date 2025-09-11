// src/api/api.js
import axios from "axios";

/* =========================================
 * Base URL & axios client
 * =======================================*/
let API_BASE =
  (import.meta?.env?.VITE_API_BASE ?? "").trim() ||
  (import.meta?.env?.VITE_API_URL ?? "").trim() ||
  (globalThis?.process?.env?.REACT_APP_API_URL ?? "").trim() ||
  (globalThis?.__API_BASE__ ?? "").trim();

// If nothing provided, default to same host + :8000 (works after IP changes)
if (!API_BASE) {
  const proto = (globalThis?.location?.protocol || "http:").replace(/:$/, "");
  const host = globalThis?.location?.hostname || "127.0.0.1";
  const port = (import.meta?.env?.VITE_API_PORT ?? "8000").toString().trim();
  API_BASE = `${proto}://${host}:${port}`;
}

console.log("[%capi", "color:#0bf;font-weight:bold", "] BASE =", API_BASE);

export const getApiBase = () => API_BASE;
export const setApiBase = (url) => {
  API_BASE = (url ?? "").trim();
  http.defaults.baseURL = API_BASE;
};

const http = axios.create({
  baseURL: API_BASE,
  timeout: 2000000,
  withCredentials: false,
  transformResponse: [
    (data) => {
      try {
        return typeof data === "string" && data ? JSON.parse(data) : data;
      } catch {
        return data;
      }
    },
  ],
  validateStatus: (s) => s >= 200 && s < 300,
});

http.interceptors.request.use((c) => c, (e) => Promise.reject(e));

/** Better error messages (unwrap FastAPI + proxied Ollama text/JSON) */
http.interceptors.response.use(
  (r) => r,
  (e) => {
    const res = e?.response;
    const data = res?.data;

    let detail = data?.detail ?? data?.error ?? data?.message ?? data ?? e?.message;

    if (typeof detail === "string") {
      const s = detail.trim();
      if (s.startsWith("{") || s.startsWith("[")) {
        try {
          const inner = JSON.parse(s);
          detail = inner?.detail ?? inner?.error ?? inner?.message ?? s;
        } catch {
          /* keep original string */
        }
      }
    }

    let msg = "";
    if (Array.isArray(detail)) {
      msg = detail.map((it) => it?.msg || it?.detail || JSON.stringify(it)).join("; ");
    } else if (detail && typeof detail === "object") {
      msg =
        detail?.msg ||
        detail?.detail ||
        detail?.error ||
        detail?.message ||
        JSON.stringify(detail);
    } else if (typeof detail === "string") {
      msg = detail;
    } else if (e?.message) {
      msg = e.message;
    } else {
      msg = "Request failed";
    }

    if (res?.status && !String(msg).startsWith(String(res.status))) {
      msg = `${res.status} · ${msg}`;
    }
    return Promise.reject(new Error(msg));
  }
);

export const httpClient = http;

/* =========================================
 * Helpers
 * =======================================*/
const toDataUrl = (b64, mime = "image/png") => (b64 ? `data:${mime};base64,${b64}` : "");

/** Join API base + relative path safely (handles missing/extra slashes) */
export const vizImageUrl = (apiPath) => {
  if (!apiPath) return "";
  if (/^(data:|https?:\/\/)/i.test(apiPath)) return apiPath;
  const b = getApiBase();
  if (!b) return apiPath;
  const base = b.replace(/\/+$/, "");
  const path = String(apiPath).replace(/^\/+/, "");
  return `${base}/${path}`;
};

/** Build absolute API URL for any relative backend path */
export const buildApiUrl = (apiPath) => vizImageUrl(apiPath);

export const vizThumbUrl = (idOrPath) =>
  vizImageUrl(
    typeof idOrPath === "string" && !/[/.]/.test(idOrPath)
      ? `/api/visualizations/${idOrPath}/thumb`
      : idOrPath
  );

export const vizTableUrl = (idOrPath) =>
  vizImageUrl(
    typeof idOrPath === "string" && !/[/.]/.test(idOrPath)
      ? `/api/visualizations/${idOrPath}/table`
      : idOrPath
  );

const withAbsUrls = (items = []) =>
  items.map((m) => ({
    ...m,
    image_url: vizImageUrl(m.image_url),
    thumb_url: vizImageUrl(m.thumb_url),
    table_csv_url: vizImageUrl(m.table_csv_url),
  }));

const extOf = (name = "") => {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
};

const lastSegment = (p = "") => {
  let s = String(p || "").trim();
  s = s.replace(/^["']|["']$/g, "");
  s = s.split(/[?#]/)[0];
  const seg = s.split(/[\\/]/).pop();
  return (seg || s).trim().replace(/^["']|["']$/g, "");
};

const ensureAllowed = (fileOrName, allowed, errMsg) => {
  let name = "";
  if (typeof fileOrName === "string") {
    name = lastSegment(fileOrName);
  } else {
    name = (fileOrName && fileOrName.name) || "";
  }
  const ext = extOf(name);
  if (!allowed.has(ext)) {
    throw new Error(`${errMsg} (got: "${name || "unknown"}")`);
  }
  return true;
};

const isFile = (v) => typeof File !== "undefined" && v instanceof File;
const isFormData = (v) => typeof FormData !== "undefined" && v instanceof FormData;

// Parse a filename from Content-Disposition header
const filenameFromContentDisposition = (cd = "", fallback = "download.bin") => {
  try {
    const mStar = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(cd);
    if (mStar?.[1]) return decodeURIComponent(mStar[1]);
    const m = /filename\s*=\s*"?([^"]+)"?/i.exec(cd);
    if (m?.[1]) return m[1];
  } catch {}
  return fallback;
};

/* =========================================
 * Allowed extensions (must match backend)
 * =======================================*/
const CHAT_DOC_EXTS = new Set([
  ".pdf",
  ".doc",
  ".docx",
  ".xls",
  ".xlsx",
  ".csv",
  ".png",
  ".jpg",
  ".jpeg",
]);

const VIZ_DATA_EXTS = new Set([".xlsx", ".xls", ".csv"]);

// Expanded to match backend and mobile captures
const IMG_EXTS = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".gif",
  ".bmp",
  ".tiff",
  ".tif",
]);

// Vision also accepts PDF (backend can convert first page → PNG)
const VISION_EXTS = new Set([...IMG_EXTS, ".pdf"]);

/* =========================================
 * Normalizer for /api/* answers
 * =======================================*/
const pickPlotUrl = (obj) => {
  if (!obj || typeof obj !== "object") return "";
  if (obj.plot_image_url) return vizImageUrl(obj.plot_image_url);
  if (obj.image_url) return vizImageUrl(obj.image_url);
  if (obj.meta?.image_url) return vizImageUrl(obj.meta.image_url);
  if (obj.image_base64) return toDataUrl(obj.image_base64);
  if (obj.meta?.image_base64) return toDataUrl(obj.meta.image_base64);
  return "";
};

const normalizeAnswer = (res) => {
  if (!res) return { answer: "❌ No answer returned." };

  const answer =
    typeof res === "string" ? res : res.answer ?? res.text ?? "❌ No answer returned.";

  const plotImageUrl = pickPlotUrl(res);
  const usedDocIds = res?.used_doc_ids || res?.used_docs || null;

  return {
    answer,
    ...(plotImageUrl ? { plotImageUrl } : {}),
    ...(usedDocIds ? { usedDocIds } : {}),
  };
};

/* =========================================
 * DOCS Q&A (Vectorstore) — always /api/ask
 * =======================================*/

const DOC_ASK_PATH = "/api/ask";

export const askDocs = async ({ question, chatId, documentId, docIds }) => {
  const payload = {
    question: (question ?? "").trim(),
    chat_id: chatId ?? null,
  };

  if (Array.isArray(docIds) && docIds.length > 0) {
    payload.doc_ids = docIds;               // vectorstore doc IDs
  } else if (documentId && documentId !== "combine") {
    payload.document_id = documentId;       // single vectorstore doc ID
  }

  const { data } = await http.post(DOC_ASK_PATH, payload, {
    headers: { "Content-Type": "application/json" },
  });
  return normalizeAnswer(data);
};

/** Back-compat wrapper so older calls still work (also forwards combineDocs → doc_ids) */
export const askQuestion = async (args = {}) => {
  const { question, chatId, documentId, docIds, combineDocs } = args || {};
  return askDocs({ question, chatId, documentId, docIds: docIds || combineDocs });
};

/* =========================================
 * Multi-file upload tied to a chat (new endpoint)
 * =======================================*/
export const uploadToChat = async (chatId, files = []) => {
  if (!chatId) throw new Error("chatId is required.");
  if (!Array.isArray(files) || files.length === 0) throw new Error("No files selected.");

  const fd = new FormData();
  files.forEach((f) => {
    ensureAllowed(
      f,
      CHAT_DOC_EXTS,
      "Only PDF, Word, Excel/CSV, or image files are allowed (.pdf, .doc, .docx, .xls, .xlsx, .csv, .png, .jpg, .jpeg)."
    );
    fd.append("files", f, f.name);
  });

  const { data } = await http.post(`/api/chats/${encodeURIComponent(chatId)}/upload`, fd);
  return data; // { chat_id, docs: [{doc_id, file_name, ...}] }
};

/* List docs for a chat (from manifest) — new endpoint */
export const listChatDocs = async (chatId) => {
  const { data } = await http.get(`/api/chats/${encodeURIComponent(chatId)}/docs`);
  return data; // { chat_id, docs: [...] }
};

/* ===== Legacy single-file upload (kept for compatibility) ===== */
export const uploadDocument = async (formData) => {
  const file = formData?.get?.("file");
  if (!file) throw new Error("No file selected.");
  ensureAllowed(
    file,
    CHAT_DOC_EXTS,
    "Only PDF, Word, Excel/CSV, or image files are allowed (.pdf, .doc, .docx, .xls, .xlsx, .csv, .png, .jpg, .jpeg)."
  );
  const chatId = formData.get("chat_id");
  if (!chatId) throw new Error("Chat ID is required.");

  const { data: uploadData } = await http.post("/api/upload/upload_file", formData);

  const { task_id, status } = uploadData;
  if (status !== "processing" || !task_id) {
    throw new Error("Upload initiation failed or no task ID returned.");
  }

  const checkStatus = async (taskId) => {
    const response = await http.get(`/api/status/${taskId}`);
    return response.data;
  };

  return new Promise((resolve, reject) => {
    const pollStatus = async () => {
      let status = await checkStatus(task_id);
      if (status.status === "processing") {
        setTimeout(pollStatus, 2000);
      } else if (status.status === "ready") {
        resolve({
          ...uploadData,
          status: "ready",
          document_id: status.document_id,
          vectorstore_path: status.vectorstore_path,
        });
      } else if (status.status === "failed") {
        reject(new Error(status.error || "Vectorstore creation failed"));
      } else {
        reject(new Error("Unknown task status"));
      }
    };
    pollStatus();
  });
};

// Renamed from listDocuments for consistency (legacy)
export const listDocuments = async (chatId) => {
  const { data } = await http.get("/api/list_documents", { params: { chat_id: chatId } });
  return data;
};

export const health = async () => {
  const { data } = await http.get("/api/health");
  return data;
};

// Extra: Ollama reachability via backend
export const ollamaHealth = async () => {
  const { data } = await http.get("/api/health/ollama");
  return data; // {ok: boolean, status?: number, error?: string}
};

/* =========================================
 * Visualization (Excel/CSV) — NEW ROUTES
 * =======================================*/

/** Upload a data file once (if you keep this UX) */
export const excelUpload = async (file, chatId) => {
  if (!file) throw new Error("Please choose a file.");
  ensureAllowed(file, VIZ_DATA_EXTS, "Only Excel/CSV files are allowed (.xlsx, .xls, .csv).");
  const fd = new FormData();
  fd.append("file", file);
  if (chatId) fd.append("chat_id", chatId);

  const { data } = await http.post("/api/excel/upload/", fd);
  return data; // { file_path, chat_id, message }
};

export const excelList = async (chatId) => {
  const { data } = await http.get("/api/excel/list", { params: { chat_id: chatId } });
  return data; // { files: [...] }
};

/** Single-file viz (ALWAYS uses the selected file the user picked) */
export const vizGenerate = async ({ file, filePath, question, title, chatId }) => {
  const fd = new FormData();
  if (file) {
    ensureAllowed(file, VIZ_DATA_EXTS, "Only Excel/CSV files are allowed (.xlsx, .xls, .csv).");
    fd.append("file", file);
  }
  if (filePath) {
    ensureAllowed(lastSegment(filePath), VIZ_DATA_EXTS, "file_path must be .xlsx, .xls, or .csv");
    fd.append("file_path", filePath);
  }
  fd.append("question", question);
  if (title) fd.append("title", title);
  if (chatId) fd.append("chat_id", chatId);

  const { data } = await http.post("/api/visualizations/generate", fd);
  if (data && typeof data === "object") {
    data.image_url = vizImageUrl(data.image_url);
    data.thumb_url = vizImageUrl(data.thumb_url);
    data.table_csv_url = vizImageUrl(data.table_csv_url);
  }
  return data;
};

/** Legacy concat combine (kept, but now calls new route) */
export const vizGenerateCombined = async ({ filePaths = [], question, title, chatId }) => {
  if (!Array.isArray(filePaths) || filePaths.length < 2) {
    throw new Error("At least two files are required to combine.");
  }
  // basic extension guard
  filePaths.forEach((p) =>
    ensureAllowed(lastSegment(p), VIZ_DATA_EXTS, "file_paths must be .xlsx, .xls, or .csv")
  );

  const fd = new FormData();
  fd.append("question", question);
  if (title) fd.append("title", title);
  if (chatId) fd.append("chat_id", chatId);
  fd.append("file_paths", JSON.stringify(filePaths)); // backend accepts JSON list string

  const { data } = await http.post("/api/visualizations/generate-combined", fd);
  if (data && typeof data === "object") {
    data.image_url = vizImageUrl(data.image_url);
    data.thumb_url = vizImageUrl(data.thumb_url);
    data.table_csv_url = vizImageUrl(data.table_csv_url);
  }
  return data;
};

/** NEW: mapped combine (two selected files + explicit mapping spec) */
export const vizGenerateCombinedMapped = async ({
  question,
  file1Path,
  file2Path,
  spec,        // { group_by:{source,column}, measure:{source,column,agg}, join:{left:{},right:{},how}, sheets?:{file1,file2} }
  title,
  chatId,
  file1,       // optional File upload instead of path
  file2,       // optional File upload instead of path
}) => {
  const fd = new FormData();
  fd.append("question", question);
  fd.append("spec_json", JSON.stringify(spec || {}));
  if (title) fd.append("title", title);
  if (chatId) fd.append("chat_id", chatId);

  if (file1) {
    ensureAllowed(file1, VIZ_DATA_EXTS, "Only Excel/CSV files are allowed.");
    fd.append("file1", file1);
  } else if (file1Path) {
    ensureAllowed(lastSegment(file1Path), VIZ_DATA_EXTS, "file1_path must be .xlsx, .xls, or .csv");
    fd.append("file1_path", file1Path);
  } else {
    throw new Error("file1 or file1Path is required.");
  }

  if (file2) {
    ensureAllowed(file2, VIZ_DATA_EXTS, "Only Excel/CSV files are allowed.");
    fd.append("file2", file2);
  } else if (file2Path) {
    ensureAllowed(lastSegment(file2Path), VIZ_DATA_EXTS, "file2_path must be .xlsx, .xls, or .csv");
    fd.append("file2_path", file2Path);
  } else {
    throw new Error("file2 or file2Path is required.");
  }

  const { data } = await http.post("/api/visualizations/generate-combined-mapped", fd);
  if (data && typeof data === "object") {
    data.image_url = vizImageUrl(data.image_url);
    data.thumb_url = vizImageUrl(data.thumb_url);
    data.table_csv_url = vizImageUrl(data.table_csv_url);
  }
  return data;
};

/** Convenience wrappers for old callers (kept; routed to new endpoints) */
export const excelPlot = async (filePath, question, title, chatId) =>
  vizGenerate({ filePath, question, title, chatId });

export const excelPlotCombine = async (filePaths = [], question, title, chatId) =>
  vizGenerateCombined({ filePaths, question, title, chatId });

/** List + helpers */
export const vizList = async ({ chatId, q, limit, offset, order } = {}) => {
  const params = {};
  if (chatId) params.chat_id = chatId;
  if (q) params.q = q;
  if (limit != null) params.limit = limit;
  if (offset != null) params.offset = offset;
  if (order) params.order = order;

  const { data } = await http.get("/api/visualizations/list", { params });

  const items = Array.isArray(data) ? data : data?.items || [];
  const total = Array.isArray(data) ? items.length : data?.total ?? items.length;
  const chatIds = Array.isArray(data) ? [] : data?.chat_ids || [];

  return { items: withAbsUrls(items), total, chatIds };
};

/* =========================================
 * Visualization Q&A (Excel/CSV)
 * =======================================*/
export const askViz = async ({ question, chatId, fileId, fileName, filePath }) => {
  const payload = {
    question: (question ?? "").trim(),
    chat_id: chatId ?? null,
    file_id: fileId ?? null,
    file_name: fileName ?? null,
    file_path: filePath ?? null,
  };
  const { data } = await http.post("/api/viz/ask", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return normalizeAnswer(data);
};

// Viz chats (sessions for Visualization tab)
export const vizChatsList = async () => {
  const { data } = await http.get("/api/excel/chats");
  return data?.chats || [];
};

export const vizChatCreate = async ({ chatName, file }) => {
  if (!file) throw new Error("Please choose an Excel/CSV file.");
  ensureAllowed(file, VIZ_DATA_EXTS, "Only Excel/CSV files are allowed (.xlsx, .xls, .csv).");
  const fd = new FormData();
  fd.append("chat_name", chatName);
  fd.append("file", file);
  const { data } = await http.post("/api/excel/chats", fd);
  return data; // { chat_id, chat_name, files, created_at }
};

/* =========================================
 * Generic chat image upload (previews)
 * =======================================*/
export const chatUploadImages = async ({ chatId, text = "", files = [] }) => {
  if (!chatId) throw new Error("chatId is required.");
  const fd = new FormData();
  fd.append("chat_id", chatId);
  fd.append("text", text);
  if (Array.isArray(files)) {
    files.forEach((f) => {
      // allow PDFs too (for scanned business cards)
      ensureAllowed(
        f,
        VISION_EXTS,
        "Only image/PDF files are allowed (.png, .jpg, .jpeg, .webp, .gif, .bmp, .tiff, .tif, .pdf)."
      );
      fd.append("files", f, f.name);
    });
  }
  const { data } = await http.post("/api/chat", fd);

  // Normalize both shapes: attachments[] and/or image_urls[]
  const attachments = (data.attachments || []).map((a) => ({
    ...a,
    url: vizImageUrl(a.url),
  }));

  const image_urls = (data.image_urls || []).map((u) => vizImageUrl(u));

  return {
    ...data,
    attachments,
    image_urls,
  };
};

export const chatHistory = async (chatId) => {
  const { data } = await http.get("/api/chat/history", { params: { chat_id: chatId } });
  const msgs = Array.isArray(data?.messages) ? data.messages : [];
  return msgs.map((m) => ({
    ...m,
    attachments: (m.attachments || []).map((a) => ({ ...a, url: vizImageUrl(a.url) })),
  }));
};

/* =========================================
 * Vision endpoint — /api/ask-image (fallback)
 * =======================================*/

// helper: default prompt that handles unknown front/back order for 2 images
const makeDefaultVisionPrompt = (imageCount = 1) => {
  const COMMON = `
You are a world-class OCR system for business cards. Read the provided image(s) EXACTLY as printed.
- Auto-detect rotation/vertical text; rotate mentally if needed.
- DO NOT guess or hallucinate. If something is not visible, omit that line.
- Preserve original capitalization, punctuation, slashes and spacing in phone numbers.
- Expand label abbreviations when presenting:
  P / Ph / Tel / Off / M -> Phone
  E / Email -> Email
  W / Web / URL -> Website
- If there are multiple phone numbers, output each on its own "Phone:" line, in the same order.
- Use only plain text Markdown (NO code fences).
`.trim();

  if (imageCount <= 1) {
    return `${COMMON}

Return your answer in EXACTLY this template and wording (omit any line where the value is missing):

Based on the image provided, here is the extracted text from the business card template:

Left Side (Dark Background):
Company Name: <as printed>
Slogan: <as printed>

Right Side (White Background):
Name: <as printed>
Position: <as printed>
Website: <as printed>
Email: <as printed>
Address: <as printed>
Phone: <as printed>
Phone: <as printed>
`.trim();
  }

  // two images; order unknown — ALWAYS output Back first, then Front
  return `${COMMON}

There are TWO images of the same card, but their upload order is UNKNOWN.

DETECTION (do not include this section in output):
- FRONT has name/title/contact cluster (phone/email/website), often with logo.
- BACK has address/tagline/legal/office info or large artwork/QR.

Return your answer in EXACTLY this structure and wording:

Of course. Here is the extracted text from the two images of the business card.

Image 1 (Back of the Card)
<write the BACK side’s lines verbatim, original order & line breaks>

Image 2 (Front of the Card)
<write the FRONT side’s lines verbatim, original order & line breaks>
`.trim();
};

export const askImage = async (arg) => {
  let fd = null;
  // default prompt (will be overridden below for 2 images)
  let prompt = "Extract key information from this image. Provide a concise summary and a JSON if possible.";
  let question = "";
  let text = "";

  const pushFirstAsFrontAliases = (f) => {
    fd.append("front_image", f, f.name);
    fd.append("frontFile", f, f.name);
    fd.append("file", f, f.name);
    fd.append("files", f, f.name);
  };
  const pushSecondAsBackAliases = (f) => {
    fd.append("back_image", f, f.name);
    fd.append("backFile", f, f.name);
    fd.append("files", f, f.name);
  };

  if (isFormData(arg)) {
    fd = arg;
    if (!fd.has("prompt")) fd.append("prompt", prompt);
  } else if (isFile(arg)) {
    ensureAllowed(arg, VISION_EXTS, "Only images or PDF are allowed for vision.");
    fd = new FormData();
    fd.append("prompt", makeDefaultVisionPrompt(1));
    fd.append("images", arg, arg.name);
    fd.append("images[]", arg, arg.name);
    pushFirstAsFrontAliases(arg);
  } else if (arg && typeof arg === "object") {
    const { images, frontFile, backFile = null, prompt: p, question: q, text: t, order = "auto" } = arg;
    if (p) prompt = String(p);
    if (q) question = String(q);
    if (t) text = String(t);

    fd = new FormData();

    if (Array.isArray(images) && images.length) {
      if (!p) prompt = makeDefaultVisionPrompt(images.length);
      fd.append("prompt", prompt);

      images.forEach((f) => {
        ensureAllowed(f, VISION_EXTS, "Only images or PDF are allowed for vision.");
        fd.append("images", f, f.name);
        fd.append("images[]", f, f.name);
      });

      pushFirstAsFrontAliases(images[0]);
      if (images[1]) pushSecondAsBackAliases(images[1]);
      if (images.length >= 2 && order === "auto") {
        fd.append("order_unknown", "true");
        fd.append("detect_front_back", "true");
      }
    } else if (frontFile || backFile) {
      if (!p) prompt = makeDefaultVisionPrompt(frontFile && backFile ? 2 : 1);
      fd.append("prompt", prompt);

      if (frontFile) {
        ensureAllowed(frontFile, VISION_EXTS, "Only images or PDF are allowed for vision.");
        pushFirstAsFrontAliases(frontFile);
        fd.append("images", frontFile, frontFile.name);
        fd.append("images[]", frontFile, frontFile.name);
      }
      if (backFile) {
        ensureAllowed(backFile, VISION_EXTS, "Only images or PDF are allowed for vision.");
        pushSecondAsBackAliases(backFile);
        fd.append("images", backFile, backFile.name);
        fd.append("images[]", backFile, backFile.name);
        if (order === "auto") {
          fd.append("order_unknown", "true");
          fd.append("detect_front_back", "true");
        }
      }
    } else {
      throw new Error("Provide images[], or frontFile/backFile, or a single File.");
    }
  } else {
    throw new Error("Invalid argument for askImage");
  }

  if (!fd.has("question") && question) fd.append("question", question);
  if (!fd.has("text") && (text || question)) fd.append("text", text || question);

  const { data } = await http.post("/api/ask-image", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  const normalizedAnswer =
    data?.data?.whatsapp ?? data?.answer ?? data?.text ?? "";

  return {
    status: data?.status ?? "ok",
    data: {
      whatsapp: normalizedAnswer,
      json: data?.data?.json ?? null,
      image_urls: (data?.data?.image_urls || []).map((u) => vizImageUrl(u)),
    },
    answer: normalizedAnswer,
    meta: data?.meta || {},
    session_id: data?.session_id || null,
  };
};

/* =========================================
 * Gemini Business Card extraction + vCard
 * =======================================*/

/**
 * Extract structured contact info from a business card image.
 * NEW: accepts `chatId` so rows are tracked per-chat for export.
 */
export const extractBusinessCard = async ({
  file,
  returnVcard = true,
  prompt = "",
  chatId = null,
  chat_id = null, // alias accepted
}) => {
  if (!file) throw new Error("Please choose an image/PDF of the business card.");
  ensureAllowed(
    file,
    VISION_EXTS,
    "Only image/PDF files are allowed (.png, .jpg, .jpeg, .webp, .gif, .bmp, .tiff, .tif, .pdf)."
  );
  const fd = new FormData();
  fd.append("file", file);
  if (prompt) fd.append("prompt", String(prompt));
  const cid = chatId || chat_id;
  if (cid) fd.append("chat_id", String(cid));

  const qs = new URLSearchParams();
  qs.set("return_vcard", returnVcard ? "true" : "false");
  if (cid) qs.set("chat_id", String(cid)); // backend can accept either form-data or query

  const { data } = await http.post(`/api/cards/extract?${qs.toString()}`, fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (data?.vcard_url) data.vcard_url = vizImageUrl(data.vcard_url);
  if (data?.vcard_url && !data?.vcardUrl) data.vcardUrl = data.vcard_url;

  return data;
};

/** Convenience alias so callers can do cardsExtract(file) or cardsExtract({file}) */
export const cardsExtract = async (arg) => {
  if (arg instanceof File) return extractBusinessCard({ file: arg, returnVcard: true });
  if (arg && typeof arg === "object" && arg.file) return extractBusinessCard(arg);
  throw new Error("cardsExtract: pass a File or { file, returnVcard? }");
};

/**
 * Build a .vcf from a JSON "card" (e.g., after user edits).
 */
export const vcardFromJson = async (card) => {
  if (!card || typeof card !== "object") throw new Error("card JSON is required.");
  const { data } = await http.post("/api/cards/from-json", { card }, {
    headers: { "Content-Type": "application/json" },
  });
  if (data?.vcard_url) data.vcard_url = vizImageUrl(data.vcard_url);
  return data;
};

/** NEW: list extracted cards for a chat (robust to either route) */
export const cardsList = async (chatId) => {
  if (!chatId) throw new Error("chatId is required.");
  try {
    const { data } = await http.get(`/api/chats/${encodeURIComponent(chatId)}/cards`);
    return data?.items || data?.cards || data || [];
  } catch {
    const { data } = await http.get("/api/cards/list", { params: { chat_id: chatId } });
    return data?.items || data?.cards || data || [];
  }
};

/** Build absolute export URL (used by both open + blob flows) */
export const cardsExportUrl = ({ chatId, format = "xlsx" }) => {
  if (!chatId) throw new Error("chatId is required.");
  const fmt = String(format || "xlsx").toLowerCase();
  const allowed = new Set(["xlsx", "csv", "vcf", "zip"]);
  if (!allowed.has(fmt)) throw new Error('format must be one of: "xlsx", "csv", "vcf", "zip".');

  return buildApiUrl(
    `/api/chats/${encodeURIComponent(chatId)}/cards/export?format=${encodeURIComponent(fmt)}`
  );
};

/** One-liner to open download in a new tab (forces :8000) */
export const openCardsExport = (args) => {
  const url = cardsExportUrl(args);
  window.open(url, "_blank", "noopener");
};

/** Export extracted cards for a chat (xlsx|csv|vcf|zip). Returns {blob, filename}. */
export const cardsExport = async ({ chatId, format = "xlsx" } = {}) => {
  const url = cardsExportUrl({ chatId, format }); // absolute URL points to backend
  const res = await axios.get(url, { responseType: "blob" }); // raw axios, no interceptors needed for download
  const blob = res.data;
  const cd =
    res.headers?.["content-disposition"] ||
    res.headers?.get?.("content-disposition");
  const fallbackName = `business-cards-${chatId}-${new Date()
    .toISOString()
    .slice(0, 10)}.${format}`;
  const filename = filenameFromContentDisposition(cd, fallbackName);
  return { blob, filename };
};

/* =========================================
 * (Optional) JSON vision ask using saved static URLs
 * =======================================*/
export const visionAsk = async ({ prompt, imageUrls = [], sessionId = null }) => {
  const payload = {
    prompt: (prompt ?? "").trim(),
    image_urls: imageUrls,
  };
  if (sessionId) payload.session_id = sessionId;

  const { data } = await http.post("/api/vision/ask", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
};

/* =========================================
 * DB: chat sessions + messages persistence
 * =======================================*/

/**
 * Ensure a chat row exists (source: "chat" | "viz").
 * Hits backend that stores into chat_sessions table.
 */
export const dbEnsureChat = async ({
  chatId,
  source = "chat",
  name = null,
  createdAt = null,
} = {}) => {
  if (!chatId) throw new Error("dbEnsureChat: chatId is required.");
  const payload = {
    chat_id: String(chatId),
    source,
    name,
    created_at: createdAt || new Date().toISOString(),
  };
  const { data } = await http.post("/api/db/chat/ensure", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
};

/**
 * Append one message (role: "user" | "ai") into chat_messages.
 * Extras supported: imageUrl, imageUrls[], tableCsvUrl, kind
 */
export const dbAppendChatMessage = async ({
  chatId,
  source = "chat",
  role,
  text = "",
  imageUrl,
  imageUrls,
  tableCsvUrl,
  kind,
  timestamp, // optional ISO string
} = {}) => {
  if (!chatId) throw new Error("dbAppendChatMessage: chatId is required.");
  if (!role) throw new Error("dbAppendChatMessage: role is required.");
  const payload = {
    chat_id: String(chatId),
    source,
    role,
    text: String(text || ""),
  };
  if (imageUrl) payload.image_url = imageUrl;
  if (Array.isArray(imageUrls) && imageUrls.length) payload.image_urls = imageUrls;
  if (tableCsvUrl) payload.table_csv_url = tableCsvUrl;
  if (kind) payload.kind = kind;
  if (timestamp) payload.timestamp = timestamp;

  const { data } = await http.post("/api/db/chat/append", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
};

/**
 * Optional bulk save: send entire chat session payload to backend to persist.
 * Mirrors DBHandler.save_full_chat_dump
 */
export const dbSaveFullChatDump = async ({ session, source = "chat" } = {}) => {
  if (!session || !session.id) throw new Error("dbSaveFullChatDump: session is required.");
  const { data } = await http.post(
    "/api/db/chat/dump",
    { source, session },
    { headers: { "Content-Type": "application/json" } }
  );
  return data;
};

/**
 * Save/Upsert a row in `documents` table.
 * Backend expects: { id?, file_name, file_path }
 * If `id` is omitted/null, backend can derive it from filename.
 */
export const saveDocumentRecord = async ({ id = null, fileName, filePath }) => {
  if (!fileName) throw new Error("saveDocumentRecord: fileName is required.");
  if (!filePath) throw new Error("saveDocumentRecord: filePath is required.");
  const payload = {
    id,
    file_name: String(fileName),
    file_path: String(filePath),
  };
  const { data } = await http.post("/api/db/documents/insert", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return data;
};
