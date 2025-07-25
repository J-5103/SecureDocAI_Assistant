import axios from 'axios';
import axios from "axios";
import { BASE_URL } from "./config"; 
const BASE_URL = 'http://192.168.0.109:8000'; // Replace with your backend IP if different

// Ask a question to a specific document
export const askQuestion = async (chatId, documentName, question) => {
  try {
    const response = await axios.post(`${BASE_URL}/document-chat`, {
      chat_id: chatId,
      document_id: documentName,  // ✅ Must be the file name like xyz.pdf
      question: question,
    });

    return response.data; // { answer }
  } catch (error) {
    console.error('❌ askQuestion API Error:', error.response?.data || error.message);
    throw error.response?.data || new Error('Unknown error in askQuestion');
  }
};


// List all uploaded documents
export const listDocuments = async () => {
  try {
    const res = await axios.get(`${BASE_URL}/list_documents`);
    return res.data;
  } catch (error) {
    console.error('❌ Error in listDocuments:', error);
    throw new Error('Failed to fetch document list.');
  }
};
