// src/components/DocumentSidebar.tsx
import { useState, useCallback, useRef, useMemo } from "react";
import {
  Upload,
  FileText,
  File,
  FileSpreadsheet,
  Trash2,
  Eye,
  ImageIcon,
  Layers,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Checkbox } from "@/components/ui/checkbox";

export interface Document {
  id: string;
  name: string;
  /** Aligned with viz types */
  type: "pdf" | "excel" | "csv" | "folder" | "other";
  size: string;
  uploadDate: string;
  status: "uploaded" | "processing" | "ready" | "failed";
  chatId: string;
  /** should equal filename (key used across the app) */
  documentId: string;
}

interface DocumentSidebarProps {
  chatId: string;
  documentList: Document[];
  /** must return a Promise so we can show loading states */
  onDocumentUpload: (files: FileList) => Promise<void>;
  onSelectDocument?: (docId: string) => void;
  selectedDocId?: string; // a specific documentId

  /** OPTIONAL: if provided, enables multi-select UX and keeps it in sync with parent */
  selectedCombineDocs?: string[];
  setSelectedCombineDocs?: React.Dispatch<React.SetStateAction<string[]>>;
}

export const DocumentSidebar = ({
  chatId,
  documentList,
  onDocumentUpload,
  onSelectDocument,
  selectedDocId,

  // optional (parent-controlled) combine selection
  selectedCombineDocs,
  setSelectedCombineDocs,
}: DocumentSidebarProps) => {
  const [dragActive, setDragActive] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [multiMode, setMultiMode] = useState<boolean>(false);
  const { toast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ---------- helpers ---------- */

  // Accept the same set you accept in the app; map to allowed doc types elsewhere.
  const SUPPORTED = useMemo(
    () => [
      ".pdf",
      ".doc",
      ".docx",
      ".xls",
      ".xlsx",
      ".csv",
      ".png",
      ".jpg",
      ".jpeg",
      ".webp",
      ".gif",
    ],
    []
  );

  const isCombineEnabled = typeof setSelectedCombineDocs === "function";

  const currentSelected = selectedCombineDocs ?? [];

  const toggleSelected = useCallback(
    (docId: string) => {
      if (!isCombineEnabled) return;
      setSelectedCombineDocs?.((prev) =>
        prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]
      );
    },
    [isCombineEnabled, setSelectedCombineDocs]
  );

  const clearSelected = useCallback(() => {
    setSelectedCombineDocs?.([]);
  }, [setSelectedCombineDocs]);

  const filterValidFiles = (fileList: FileList): FileList => {
    const valid = Array.from(fileList).filter((f) =>
      SUPPORTED.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    if (valid.length !== fileList.length) {
      toast({
        title: "Some files were skipped",
        description: `Only ${SUPPORTED.join(", ")} are supported.`,
        variant: "destructive",
      });
    }
    const dt = new DataTransfer();
    valid.forEach((f) => dt.items.add(f));
    return dt.files;
  };

  const launchFilePicker = () => fileInputRef.current?.click();

  const getFileIcon = (type: Document["type"]) => {
    const iconClasses = "w-5 h-5";
    switch (type) {
      case "pdf":
        return <FileText className={`${iconClasses} text-red-600`} />;
      case "excel":
        return <FileSpreadsheet className={`${iconClasses} text-green-600`} />;
      case "csv":
        return <FileSpreadsheet className={`${iconClasses} text-emerald-600`} />;
      case "folder":
        return <File className={`${iconClasses} text-indigo-600`} />;
      case "other":
      default:
        // images, doc/docx, etc.
        return <ImageIcon className={`${iconClasses} text-purple-600`} />;
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

  /* ---------- upload handlers ---------- */

  const performUpload = async (files: FileList) => {
    if (!files.length) {
      toast({
        title: "Upload Error",
        description: "No valid files detected.",
        variant: "destructive",
      });
      return;
    }
    setIsUploading(true);
    toast({ title: "Upload started", description: `${files.length} file(s) are being processed.` });

    try {
      await onDocumentUpload(files);
      toast({
        title: "Upload completed",
        description: `${files.length} file(s) processed successfully.`,
      });
    } catch (err: any) {
      toast({
        title: "Upload failed",
        description: err?.message || "An error occurred during upload.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = e.dataTransfer?.files;
    if (files?.length) performUpload(filterValidFiles(files));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files?.length) performUpload(filterValidFiles(files));
    e.target.value = "";
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /* ---------- render ---------- */

  return (
    <aside className="w-80 bg-white border-r border-gray-200 flex flex-col">
      {/* Uploader */}
      <div className="p-4 border-b border-gray-200">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-200 ${
            dragActive
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-blue-300 hover:bg-blue-50"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(true);
          }}
          onDragEnter={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setDragActive(false);
          }}
          onDrop={handleDrop}
        >
          <Upload
            className={`w-8 h-8 mx-auto mb-3 ${
              dragActive ? "text-blue-500" : "text-gray-400"
            }`}
          />
          <p className="text-sm font-medium mb-2 text-gray-800">
            Drop files here or{" "}
            <button
              type="button"
              onClick={launchFilePicker}
              className="text-blue-600 cursor-pointer hover:underline"
            >
              browse
            </button>
            {isUploading && (
              <span className="ml-2 text-blue-600 animate-pulse">Processing…</span>
            )}
          </p>
          <p className="text-xs text-gray-500">
            PDF, Word, Excel/CSV, and image files supported
          </p>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={SUPPORTED.join(",")}
            className="hidden"
            onChange={handleFileInput}
            disabled={isUploading}
          />

          <button
            onClick={launchFilePicker}
            className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-blue-400"
            aria-label="Upload files"
            disabled={isUploading}
          >
            Upload
          </button>
        </div>
      </div>

      {/* List header / multi-select toggles */}
      <div className="px-4 pt-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">
            Files ({documentList.length})
          </h3>

          {isCombineEnabled && (
            <button
              type="button"
              onClick={() => {
                const next = !multiMode;
                setMultiMode(next);
                if (!next) clearSelected();
              }}
              className={`inline-flex items-center gap-1 text-xs px-2 py-1 rounded border ${
                multiMode
                  ? "bg-blue-50 text-blue-700 border-blue-200"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
              }`}
              title="Toggle multi-select"
            >
              <Layers className="w-3.5 h-3.5" />
              {multiMode ? "Multi-select ON" : "Multi-select"}
            </button>
          )}
        </div>

        {isCombineEnabled && multiMode && (
          <div className="mt-2 mb-1 flex items-center justify-between">
            <div className="text-xs text-gray-600">
              Selected: <span className="font-medium">{currentSelected.length}</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="text-xs text-gray-600 hover:text-gray-800 underline"
                onClick={clearSelected}
              >
                Clear
              </button>
            </div>
          </div>
        )}
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 pt-2">
          {/* Files */}
          <div className="space-y-2 mt-2">
            {documentList.map((doc) => {
              const isSelected = selectedDocId === doc.documentId;
              const disabled = doc.status === "processing";
              const checked = isCombineEnabled && currentSelected.includes(doc.documentId);

              return (
                <div
                  key={doc.id}
                  className={`group p-3 rounded-lg transition-all ${
                    isSelected ? "bg-gray-50 ring-1 ring-gray-200" : "bg-gray-50 hover:bg-gray-100"
                  } ${disabled ? "opacity-60 pointer-events-none" : "cursor-pointer"}`}
                  onClick={() => {
                    if (multiMode && isCombineEnabled) {
                      toggleSelected(doc.documentId);
                      return;
                    }
                    onSelectDocument?.(doc.documentId);
                  }}
                  title={doc.name}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      if (multiMode && isCombineEnabled) toggleSelected(doc.documentId);
                      else onSelectDocument?.(doc.documentId);
                    }
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1 min-w-0">
                      {getFileIcon(doc.type)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate text-gray-800">
                          {doc.name}
                        </p>
                        <p className="text-xs text-gray-500">
                          {doc.size} •{" "}
                          {new Date(doc.uploadDate).toLocaleString("en-IN", {
                            timeZone: "Asia/Kolkata",
                          })}
                        </p>
                        <span
                          className={`inline-block mt-1 px-2 py-0.5 text-xs rounded ${getStatusColor(
                            doc.status
                          )}`}
                          role="status"
                        >
                          {doc.status}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      {isCombineEnabled && multiMode && (
                        <Checkbox
                          checked={checked}
                          onCheckedChange={() => toggleSelected(doc.documentId)}
                          className="border-gray-300"
                          aria-label={checked ? "Unselect" : "Select"}
                        />
                      )}
                      <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button
                          className="p-1 hover:bg-blue-100 rounded text-blue-600"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (multiMode && isCombineEnabled) {
                              toggleSelected(doc.documentId);
                            } else {
                              onSelectDocument?.(doc.documentId);
                            }
                          }}
                          aria-label="Preview / focus"
                          title="Preview / focus"
                          disabled={disabled}
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          className="p-1 hover:bg-red-100 rounded text-red-600"
                          onClick={(e) => {
                            e.stopPropagation();
                            toast({
                              title: "Not implemented",
                              description: "Delete functionality not implemented yet.",
                            });
                          }}
                          aria-label="Delete"
                          title="Delete"
                          disabled={disabled}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
            {documentList.length === 0 && (
              <p className="text-xs text-gray-500">No files uploaded yet.</p>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
};

export default DocumentSidebar;
