// src/api/document.api.js
import { httpClient as http } from "./api";

export const uploadExcel = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await http.post("/api/excel/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data; // { file_path, message }
};

export const askExcelQuestion = async (filePath, question) => {
  const formData = new FormData();
  formData.append("file_path", filePath);
  formData.append("question", question);
  const { data } = await http.post("/api/excel/ask/", formData);
  return data; // { answer }
};

export const generateExcelPlot = async (filePath, question, title) => {
  const formData = new FormData();
  formData.append("file_path", filePath);
  formData.append("question", question);
  if (title) formData.append("title", title);
  const { data } = await http.post("/api/excel/plot/", formData);
  return data; // { image_base64, meta }
};
