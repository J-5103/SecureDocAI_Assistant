import React, { useState } from "react";
import {
  Upload,
  FileText,
  File,
  FileSpreadsheet,
  Trash2,
  Eye,
  ImageIcon,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

export interface Document {
  id: string;
  name: string;
  type: "pdf" | "excel" | "word" | "image";
  size: string;
  uploadDate: string;
  status: "uploaded" | "processing" | "ready";
  chatId: string;
  documentId: string;
}

interface DocumentSidebarProps {
  chatId: string;
  documentList: Document[];
  onDocumentUpload: (files: FileList) => void;
  onSelectDocument?: (docId: string) => void;
  selectedDocId?: string;
  selectedCombineDocs: string[];
  // ✅ Type like a React state setter so functional updaters work
  setSelectedCombineDocs: React.Dispatch<React.SetStateAction<string[]>>;
}

export const DocumentSidebar = ({
  chatId,
  documentList,
  onDocumentUpload,
  onSelectDocument,
  selectedDocId,
  selectedCombineDocs,
  setSelectedCombineDocs,
}: DocumentSidebarProps) => {
  const [dragActive, setDragActive] = useState(false);
  const { toast } = useToast();

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const files = e.dataTransfer?.files;
    if (files?.length > 0) {
      const validFiles = filterValidFiles(files);
      validFiles.length ? onDocumentUpload(validFiles) : showInvalidFileToast();
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files?.length > 0) {
      const validFiles = filterValidFiles(files);
      validFiles.length ? onDocumentUpload(validFiles) : showInvalidFileToast();
    }
  };

  const filterValidFiles = (fileList: FileList): FileList => {
    const supportedTypes = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".png", ".jpg", ".jpeg"];
    const validArray = Array.from(fileList).filter(file =>
      supportedTypes.some(ext => file.name.toLowerCase().endsWith(ext))
    );
    const dataTransfer = new DataTransfer();
    validArray.forEach(file => dataTransfer.items.add(file));
    return dataTransfer.files;
  };

  const showInvalidFileToast = () => {
    toast({
      title: "Upload Error",
      description: "Only PDF, Word, Excel, and image files are supported.",
    });
  };

  const getFileIcon = (type: Document["type"]): React.ReactNode => {
    switch (type) {
      case "pdf": return <FileText className="w-5 h-5 text-red-600" />;
      case "excel": return <FileSpreadsheet className="w-5 h-5 text-green-600" />;
      case "word": return <File className="w-5 h-5 text-blue-600" />;
      case "image": return <ImageIcon className="w-5 h-5 text-purple-600" />;
    }
  };

  const getStatusColor = (status: Document["status"]) => {
    switch (status) {
      case "uploaded": return "bg-gray-200 text-gray-600";
      case "processing": return "bg-yellow-200 text-yellow-800 animate-pulse";
      case "ready": return "bg-green-100 text-green-700";
    }
  };

  const handleSelectCombineDocument = (docId: string) => {
    setSelectedCombineDocs(prev =>
      prev.includes(docId)
        ? prev.filter(id => id !== docId)
        : [...prev, docId]
    );
  };

  return (
    <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-200 ${
            dragActive
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-blue-300 hover:bg-blue-50"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <Upload
            className={`w-8 h-8 mx-auto mb-3 ${dragActive ? "text-blue-500" : "text-gray-400"}`}
          />
          <p className="text-sm font-medium mb-2">
            Drop files here or{" "}
            <label className="text-blue-600 cursor-pointer hover:underline">
              browse
              <input
                type="file"
                multiple
                accept=".pdf,.doc,.docx,.xls,.xlsx,.png,.jpg,.jpeg"
                className="hidden"
                onChange={handleFileInput}
              />
            </label>
          </p>
          <p className="text-xs text-gray-500">PDF, Word, Excel, and image files supported</p>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <h3 className="text-sm font-semibold mb-3">Documents ({documentList.length})</h3>
          <div className="space-y-2">
            <div
              className={`group p-3 rounded-lg transition-all cursor-pointer border-dashed border-2 ${
                selectedDocId === "combine"
                  ? "bg-blue-50 border-blue-400"
                  : "bg-white hover:bg-gray-100 border-transparent"
              }`}
              onClick={() => onSelectDocument?.("combine")}
            >
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-indigo-600" />
                <div className="flex-1">
                  <p className="text-sm font-medium">🔗 Combine PDFs</p>
                  <p className="text-xs text-gray-500">Ask using multiple PDFs</p>
                </div>
              </div>
            </div>

            {documentList.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <FileText className="w-12 h-12 mx-auto mb-3" />
                <p className="text-sm">No documents uploaded yet</p>
              </div>
            ) : (
              documentList.map((doc) => (
                <div
                  key={doc.id}
                  className={`group p-3 rounded-lg transition-all cursor-pointer ${
                    selectedCombineDocs.includes(doc.documentId)
                      ? "bg-blue-50 border-blue-400"
                      : "bg-gray-50 hover:bg-gray-100"
                  }`}
                  onClick={() => handleSelectCombineDocument(doc.documentId)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1 min-w-0">
                      {getFileIcon(doc.type)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{doc.name}</p>
                        <p className="text-xs text-gray-500">
                          {doc.size} • {new Date(doc.uploadDate).toLocaleDateString()}
                        </p>
                        <span
                          className={`inline-block mt-1 px-2 py-0.5 text-xs rounded ${getStatusColor(doc.status)}`}
                        >
                          {doc.status}
                        </span>
                      </div>
                    </div>
                    <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button className="p-1 hover:bg-blue-100 rounded text-blue-600">
                        <Eye className="w-4 h-4" />
                      </button>
                      <button className="p-1 hover:bg-red-100 rounded text-red-600">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
