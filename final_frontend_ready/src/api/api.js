// src/api/api.js
import axios from "axios";

// --- API base config ---
let API_BASE = (import.meta?.env?.VITE_API_BASE ?? "").trim();
if (!API_BASE) API_BASE = ""; // proxy => /api/*

export const getApiBase = () => API_BASE;
export const setApiBase = (url) => {
  API_BASE = (url ?? "").trim();
  http.defaults.baseURL = API_BASE;
};

// --- Axios instance ---
const http = axios.create({
  baseURL: API_BASE,
  timeout: 200000, // Ollama can be slow
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

// --- Interceptors ---
http.interceptors.request.use(
  (config) => {
    console.log(`📥 ${config.method?.toUpperCase()} ${config.baseURL || ""}${config.url}`);
    return config;
  },
  (error) => {
    console.error("❌ Request Error:", error);
    return Promise.reject(error);
  }
);

http.interceptors.response.use(
  (res) => res,
  (error) => {
    console.error("❌ Response Error:", {
      status: error?.response?.status,
      url: error?.config?.url,
      method: error?.config?.method,
      data: error?.response?.data,
      message: error?.message,
    });
    return Promise.reject(error);
  }
);

// --- Helpers ---
const extractError = (error, fallback = "Something went wrong") =>
  error?.response?.data?.message ||
  error?.response?.data?.error ||
  error?.message ||
  fallback;

const normalizeAnswer = (res) => {
  if (res && typeof res === "object" && "answer" in res) return res;
  if (typeof res === "string") return { answer: res };
  return { answer: "❌ No answer returned." };
};

// 🧠 Ask question (single or combined docs)
export const askQuestion = async ({ question, chatId, documentId, combineDocs }) => {
  const payload = {
    question: question?.trim() || "",
    chat_id: chatId,
    document_id: documentId && documentId !== "combine" ? documentId : null,
    combine_docs: documentId === "combine" ? (combineDocs || []) : [],
  };
  console.log("📤 Sending payload to /api/ask:", payload);

  try {
    const { data } = await http.post("/api/ask", payload, {
      headers: { "Content-Type": "application/json" },
    });
    return normalizeAnswer(data);
  } catch (error) {
    throw new Error(extractError(error, "Something went wrong while asking."));
  }
};

// 📄 Upload PDF/Word/Excel/Image
export const uploadPdf = async (formData) => {
  try {
    const { data } = await http.post("/api/upload/upload_file", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  } catch (error) {
    throw new Error(extractError(error, "Upload failed"));
  }
};

// 🖼 Upload image (uses same endpoint)
export const uploadImage = async (formData) => {
  try {
    const { data } = await http.post("/api/upload/upload_file", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  } catch (error) {
    throw new Error(extractError(error, "Image upload failed"));
  }
};

// 📃 List documents by chat_id
export const listDocuments = async (chatId) => {
  try {
    const { data } = await http.get("/api/list_documents", { params: { chat_id: chatId } });
    return data;
  } catch (error) {
    throw new Error(extractError(error, "List documents failed"));
  }
};

// 🩺 Backend health check
export const health = async () => {
  try {
    const { data } = await http.get("/api/health");
    return data;
  } catch (error) {
    throw new Error(extractError(error, "Backend not reachable"));
  }
};

// Export axios instance for advanced use
export const httpClient = http;
