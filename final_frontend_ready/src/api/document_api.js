import axios from "axios";

const BASE_URL = 'http://192.168.0.109:8000';

export const uploadExcel = async (file) => {
  const formData = new FormData();
  formData.append("file", file);
  const response = await axios.post(`${BASE_URL}/excel/upload/`, formData);
  return response.data;
};

export const askExcelQuestion = async (filePath, question) => {
  const formData = new FormData();
  formData.append("file_path", filePath);
  formData.append("question", question);
  const response = await axios.post(`${BASE_URL}/excel/ask/`, formData);
  return response.data;
};

export const generateExcelPlot = async (filePath, question) => {
  const formData = new FormData();
  formData.append("file_path", filePath);
  formData.append("question", question);
  const response = await axios.post(`${BASE_URL}/excel/plot/`, formData);
  return response.data;
};