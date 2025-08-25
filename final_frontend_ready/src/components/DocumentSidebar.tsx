import { useState, useCallback, useEffect } from "react";
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
  status: "uploaded" | "processing" | "ready" | "failed";
  chatId: string;
  documentId: string;
}

interface DocumentSidebarProps {
  chatId: string;
  documentList: Document[];
  onDocumentUpload: (files: FileList) => Promise<void>; // Enforce Promise return type
  onSelectDocument?: (docId: string) => void;
  selectedDocId?: string;
  selectedCombineDocs: string[];
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
  const [isUploading, setIsUploading] = useState(false); // Track upload progress

  // Current date and time (05:17 PM IST, August 19, 2025) for initial timestamp reference
  const currentDate = new Date("2025-08-19T17:17:00+05:30").toISOString();

  const filterValidFiles = (fileList: FileList): FileList => {
    const supported = [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".png", ".jpg", ".jpeg"];
    const valid = Array.from(fileList).filter((f) =>
      supported.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (valid.length !== fileList.length) {
      toast({
        title: "Invalid Files",
        description: `Some files were skipped. Only ${supported.join(", ")} are supported.`,
        variant: "destructive",
      });
    }
    const dt = new DataTransfer();
    valid.forEach((f) => dt.items.add(f));
    return dt.files;
  };

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragActive(false);
      const files = e.dataTransfer?.files;
      if (files?.length) {
        const validFiles = filterValidFiles(files);
        if (validFiles.length > 0) {
          setIsUploading(true); // Start loading state
          onDocumentUpload(validFiles)
            .then(() => {
              setIsUploading(false); // End loading state on success
              toast({
                title: "Upload Completed",
                description: `${validFiles.length} file(s) processed successfully.`,
              });
            })
            .catch((error: Error) => {
              setIsUploading(false); // End loading state on error
              toast({
                title: "Upload Failed",
                description: error.message || "An error occurred during upload.",
                variant: "destructive",
              });
            });
          toast({
            title: "Upload Started",
            description: `${validFiles.length} file(s) being processed.`,
          });
        } else {
          toast({
            title: "Upload Error",
            description: "No valid files detected for upload.",
            variant: "destructive",
          });
        }
      }
    },
    [onDocumentUpload, toast]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files?.length) {
        const validFiles = filterValidFiles(files);
        if (validFiles.length > 0) {
          setIsUploading(true); // Start loading state
          onDocumentUpload(validFiles)
            .then(() => {
              setIsUploading(false); // End loading state on success
              toast({
                title: "Upload Completed",
                description: `${validFiles.length} file(s) processed successfully.`,
              });
            })
            .catch((error: Error) => {
              setIsUploading(false); // End loading state on error
              toast({
                title: "Upload Failed",
                description: error.message || "An error occurred during upload.",
                variant: "destructive",
              });
            });
          toast({
            title: "Upload Started",
            description: `${validFiles.length} file(s) being processed.`,
          });
        }
      }
      e.target.value = "";
    },
    [onDocumentUpload, toast]
  );

  const getFileIcon = (type: Document["type"]) => {
    const iconClasses = "w-5 h-5";
    switch (type) {
      case "pdf":
        return <FileText className={`${iconClasses} text-red-600`} />;
      case "excel":
        return <FileSpreadsheet className={`${iconClasses} text-green-600`} />;
      case "word":
        return <File className={`${iconClasses} text-blue-600`} />;
      case "image":
        return <ImageIcon className={`${iconClasses} text-purple-600`} />;
      default:
        return <File className={`${iconClasses} text-gray-600`} />;
    }
  };

  const getStatusColor = (status: Document["status"]) => {
    switch (status) {
      case "uploaded":
        return "bg-gray-200 text-gray-700";
      case "processing":
        return "bg-yellow-200 text-yellow-900 animate-pulse";
      case "ready":
        return "bg-green-100 text-green-700";
      case "failed":
        return "bg-red-100 text-red-700";
      default:
        return "bg-gray-200 text-gray-700";
    }
  };

  const handleSelectCombineDocument = useCallback(
    (docId: string) => {
      setSelectedCombineDocs((prev) =>
        prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
      );
      if (!selectedCombineDocs.includes(docId) && onSelectDocument) {
        onSelectDocument(docId);
      }
    },
    [setSelectedCombineDocs, selectedCombineDocs, onSelectDocument]
  );

  // Effect to update document list status if backend provides updates
  useEffect(() => {
    const updateStatuses = () => {
      setSelectedCombineDocs((prev) => {
        const updated = [...prev];
        documentList.forEach((doc) => {
          if (doc.status === "failed" && prev.includes(doc.documentId)) {
            const index = updated.indexOf(doc.documentId);
            if (index !== -1) updated.splice(index, 1);
          }
        });
        return updated;
      });
    };
    updateStatuses();
  }, [documentList]);

  return (
    <aside className="w-80 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-200 ${
            dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-300 hover:bg-blue-50"
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
          <p className="text-sm font-medium mb-2 text-gray-800">
            Drop files here or{" "}
            <label
              className="text-blue-600 cursor-pointer hover:underline"
            >
              browse
              <input
                type="file"
                multiple
                accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.png,.jpg,.jpeg"
                className="hidden"
                onChange={handleFileInput}
                disabled={isUploading} // Disable during upload
              />
            </label>
            {isUploading && (
              <span className="ml-2 text-blue-600 animate-pulse">Processing...</span>
            )}
          </p>
          <p className="text-xs text-gray-500">PDF, Word, Excel/CSV, and image files supported</p>
          <button
            onClick={() => {
              const input = document.querySelector('input[type="file"]');
              if (input instanceof HTMLInputElement) input.click();
            }}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-blue-400"
            aria-label="Upload files"
            disabled={isUploading} // Disable button during upload
          >
            Create
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <h3 className="text-sm font-semibold mb-3 text-gray-800">Files ({documentList.length})</h3>
          <div className="space-y-2">
            <div
              className={`group p-3 rounded-lg transition-all cursor-pointer border-dashed border-2 ${
                selectedDocId === "combine" ? "bg-blue-50 border-blue-400" : "bg-white hover:bg-gray-100 border-transparent"
              }`}
              onClick={() => onSelectDocument?.("combine")}
              role="button"
              tabIndex={0}
              onKeyPress={(e) => e.key === "Enter" && onSelectDocument?.("combine")}
            >
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-indigo-600" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-800">🔗 Combine files</p>
                  <p className="text-xs text-gray-500">Ask using multiple files</p>
                </div>
              </div>
            </div>

            {Array.isArray(documentList) && documentList.length === 0 ? (
              <div className="text-center py-8 text-gray-500" role="status">
                <FileText className="w-12 h-12 mx-auto mb-3" />
                <p className="text-sm">No files uploaded yet</p>
              </div>
            ) : Array.isArray(documentList) ? (
              documentList.map((doc) => (
                <div
                  key={doc.id}
                  className={`group p-3 rounded-lg transition-all cursor-pointer ${
                    selectedCombineDocs.includes(doc.documentId)
                      ? "bg-blue-50 border-2 border-blue-400"
                      : selectedDocId === doc.documentId
                      ? "bg-green-50 border-2 border-green-400"
                      : "bg-gray-50 hover:bg-gray-100"
                  }`}
                  onClick={() => {
                    if (!selectedCombineDocs.includes(doc.documentId)) {
                      setSelectedCombineDocs([]);
                      onSelectDocument?.(doc.documentId);
                    } else {
                      handleSelectCombineDocument(doc.documentId);
                    }
                  }}
                  title={doc.name}
                  role="button"
                  tabIndex={0}
                  onKeyPress={(e) => e.key === "Enter" && handleSelectCombineDocument(doc.documentId)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1 min-w-0">
                      {getFileIcon(doc.type)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate text-gray-800">{doc.name}</p>
                        <p className="text-xs text-gray-500">
                          {doc.size} • {new Date(doc.uploadDate).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
                        </p>
                        <span
                          className={`inline-block mt-1 px-2 py-0.5 text-xs rounded ${getStatusColor(doc.status)}`}
                          role="status"
                        >
                          {doc.status}
                        </span>
                      </div>
                    </div>

                    <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        className="p-1 hover:bg-blue-100 rounded text-blue-600"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectDocument?.(doc.documentId);
                        }}
                        aria-label="Preview / focus"
                        title="Preview / focus"
                        disabled={doc.status === "processing"} // Disable during processing
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                      <button
                        className="p-1 hover:bg-red-100 rounded text-red-600"
                        onClick={(e) => {
                          e.stopPropagation();
                          toast({
                            title: "Not Implemented",
                            description: "Delete functionality not implemented yet.",
                          });
                        }}
                        aria-label="Delete"
                        title="Delete"
                        disabled={doc.status === "processing"} // Disable during processing
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))
            ) : null}
          </div>
        </div>
      </div>
    </aside>
  );
};