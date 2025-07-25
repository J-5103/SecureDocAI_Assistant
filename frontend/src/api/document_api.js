import axios from "axios";
import { BASE_URL } from "./config"; 

export const uploadFile = async (file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${BASE_URL}/upload_pdf`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    console.error("❌ Error in uploadFile:", error);
    throw new Error("Failed to upload file.");
  }
};

