// src/api/api.js
import axios from "axios";

/* =========================================
 * Base URL & axios client
 * =======================================*/
let API_BASE = (import.meta?.env?.VITE_API_BASE ?? "").trim();
if (!API_BASE) API_BASE = "http://192.168.0.109:8000"; // fallback for local dev

export const getApiBase = () => API_BASE;
export const setApiBase = (url) => {
  API_BASE = (url ?? "").trim();
  http.defaults.baseURL = API_BASE;
};

const http = axios.create({
  baseURL: API_BASE,
  timeout: 200000,
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
http.interceptors.response.use(
  (r) => r,
  (e) => {
    const msg =
      e?.response?.data?.detail ||
      e?.response?.data?.error ||
      e?.message ||
      "Request failed";
    return Promise.reject(new Error(msg));
  }
);

export const httpClient = http;

/* =========================================
 * Helpers
 * =======================================*/
const toDataUrl = (b64, mime = "image/png") =>
  b64 ? `data:${mime};base64,${b64}` : "";

// Absolute URL helper for images returned by backend (/static/… paths)
export const vizImageUrl = (apiPath) => {
  if (!apiPath) return "";
  if (/^(data:|https?:\/\/)/i.test(apiPath)) return apiPath;
  const b = getApiBase();
  return b ? `${b.replace(/\/+$/, "")}${apiPath}` : apiPath;
};

const withAbsUrls = (items = []) =>
  items.map((m) => ({
    ...m,
    image_url: vizImageUrl(m.image_url),
    thumb_url: vizImageUrl(m.thumb_url),
  }));

// --- path/extension helpers ---
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
const IMG_EXTS = new Set([".png", ".jpg", ".jpeg", ".webp", ".gif"]);

/* =========================================
 * Normalizer for /api/ask & /api/viz/ask
 * =======================================*/
const normalizeAnswer = (res) => {
  if (!res) return { answer: "❌ No answer returned." };

  const answer =
    typeof res === "string" ? res : res.answer ?? res.text ?? "❌ No answer returned.";

  let plotImageUrl = "";
  if (typeof res.plot_image_url === "string" && res.plot_image_url) {
    plotImageUrl = vizImageUrl(res.plot_image_url);
  } else if (typeof res.image_url === "string" && res.image_url) {
    plotImageUrl = vizImageUrl(res.image_url);
  } else if (typeof res.image_base64 === "string" && res.image_base64) {
    plotImageUrl = toDataUrl(res.image_base64);
  }

  return plotImageUrl ? { answer, plotImageUrl } : { answer };
};

/* =========================================
 * Chat (PDF/Docs) — Q&A
 * =======================================*/
export const askQuestion = async ({
  question,
  chatId,
  documentId,
  combineDocs,
  intent,
}) => {
  const payload = {
    question: (question ?? "").trim(),
    chat_id: chatId,
    intent: intent || undefined,
    document_id: documentId && documentId !== "combine" ? documentId : null,
    combine_docs: Array.isArray(combineDocs) ? combineDocs : [],
  };

  const { data } = await http.post("/api/ask", payload, {
    headers: { "Content-Type": "application/json" },
  });
  return normalizeAnswer(data);
};

// Upload to Chat Sessions — supports docs/images; async vectorstore creation
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

  const { data: uploadData } = await http.post("/api/upload/upload_file", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

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

// Renamed from listDocuments for consistency
export const listDocuments = async (chatId) => {
  const { data } = await http.get("/api/list_documents", { params: { chat_id: chatId } });
  return data;
};

export const health = async () => {
  const { data } = await http.get("/api/health");
  return data;
};

/* =========================================
 * Visualization (Excel/CSV)
 * =======================================*/
export const excelUpload = async (file, chatId) => {
  if (!file) throw new Error("Please choose a file.");
  ensureAllowed(file, VIZ_DATA_EXTS, "Only Excel/CSV files are allowed (.xlsx, .xls, .csv).");
  const fd = new FormData();
  fd.append("file", file);
  if (chatId) fd.append("chat_id", chatId);
  const { data } = await http.post("/api/excel/upload/", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // { file_path, chat_id, message }
};

export const excelList = async (chatId) => {
  const { data } = await http.get("/api/excel/list", { params: { chat_id: chatId } });
  return data; // { files: [...] }
};

export const excelPlot = async (filePath, question, title, chatId) => {
  const fd = new FormData();
  if (filePath) {
    ensureAllowed(lastSegment(filePath), VIZ_DATA_EXTS, "file_path must be .xlsx, .xls, or .csv");
    fd.append("file_path", filePath);
  }
  fd.append("question", question);
  if (title) fd.append("title", title);
  if (chatId) fd.append("chat_id", chatId);

  const { data } = await http.post("/api/excel/plot/", fd);
  return {
    ...data,
    image_url: vizImageUrl(data?.image_url),
    thumb_url: vizImageUrl(data?.thumb_url),
    image_base64: data?.image_base64,
  };
};

export const excelPlotCombine = async (filePaths = [], question, title, chatId) => {
  if (!Array.isArray(filePaths) || filePaths.length < 2) {
    throw new Error("At least two files are required to combine.");
  }

  const cleaned = filePaths.map((p) => lastSegment(p));
  cleaned.forEach((name) =>
    ensureAllowed(name, VIZ_DATA_EXTS, "file_paths must be .xlsx, .xls, or .csv")
  );

  const payload = { file_paths: filePaths, question, title, chat_id: chatId };
  const { data } = await http.post("/api/excel/plot/combine", payload, {
    headers: { "Content-Type": "application/json" },
  });

  return {
    ...data,
    image_url: vizImageUrl(data?.image_url),
    thumb_url: vizImageUrl(data?.thumb_url),
    image_base64: data?.image_base64,
  };
};

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

  const { data } = await http.post("/api/visualizations/generate", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (data && typeof data === "object") {
    data.image_url = vizImageUrl(data.image_url);
    data.thumb_url = vizImageUrl(data.thumb_url);
  }
  return data;
};

// Visualization Q&A (Excel/CSV)
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
  const { data } = await http.post("/api/excel/chats", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // { chat_id, chat_name, files, created_at }
};

/* =========================================
 * NEW — Generic chat image upload (previews)
 * Saves images to /static/uploads/<chat_id>/ and returns URLs
 * =======================================*/
export const chatUploadImages = async ({ chatId, text = "", files = [] }) => {
  if (!chatId) throw new Error("chatId is required.");
  const fd = new FormData();
  fd.append("chat_id", chatId);
  fd.append("text", text);
  if (Array.isArray(files)) {
    files.forEach((f) => {
      ensureAllowed(f, IMG_EXTS, "Only image files are allowed (.png, .jpg, .jpeg, .webp, .gif).");
      fd.append("files", f, f.name);
    });
  }
  const { data } = await http.post("/api/chat", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  // data: { message_id, chat_id, text, attachments:[{filename,url}], created_at }
  return {
    ...data,
    attachments: (data.attachments || []).map((a) => ({
      ...a,
      url: vizImageUrl(a.url),
    })),
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
 * NEW — Vision endpoints
 *   1) /api/ask-image (multipart: front/back images)
 *   2) /api/vision/ask (JSON: prompt + image_urls)
 * =======================================*/

// Multipart: send one or two images for vision pipeline
export const askImage = async ({ frontFile, backFile = null, sessionId = null }) => {
  if (!frontFile) throw new Error("frontFile is required.");
  ensureAllowed(frontFile, IMG_EXTS, "Only image files are allowed (.png, .jpg, .jpeg, .webp, .gif).");
  if (backFile) ensureAllowed(backFile, IMG_EXTS, "Only image files are allowed (.png, .jpg, .jpeg, .webp, .gif).");

  const fd = new FormData();
  fd.append("front_image", frontFile);
  if (backFile) fd.append("back_image", backFile);
  if (sessionId) fd.append("session_id", sessionId);

  const { data } = await http.post("/api/ask-image", fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  // shape: { status, data: { whatsapp|json|vcard?, image_urls? }, session_id }
  return {
    ...data,
    data: {
      ...data?.data,
      image_urls: (data?.data?.image_urls || []).map((u) => vizImageUrl(u)),
    },
  };
};

// JSON: prompt + already-uploaded static URLs (from chatUploadImages)
export const visionAsk = async ({ prompt, imageUrls = [], sessionId = null }) => {
  const payload = {
    prompt: (prompt ?? "").trim(),
    image_urls: imageUrls,
    session_id: sessionId ?? null,
  };
  const { data } = await http.post("/api/vision/ask", payload, {
    headers: { "Content-Type": "application/json" },
  });
  // shape: { status, data:{ answer, raw?, model? }, session_id }
  return data;
};
