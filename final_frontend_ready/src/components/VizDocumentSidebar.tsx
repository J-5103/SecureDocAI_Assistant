// src/components/VizDocumentSidebar.tsx
import React, { useRef, useState } from "react";
import {
  Upload,
  FileText,
  FileSpreadsheet,
  Loader2,
  CheckCircle2,
  Clock,
  File,
  Folder,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface VizDocument {
  id: string;
  name: string;
  type: "pdf" | "excel" | "csv" | "folder" | "other";
  size: string;
  uploadDate: string;
  status: "uploaded" | "processing" | "ready";
  chatId: string;
  documentId: string;
}

interface VizDocumentSidebarProps {
  chatId: string; // kept for API symmetry, even if not used here
  documentList: VizDocument[];
  onDocumentUpload: (files: FileList) => Promise<void> | void;
}

export const VizDocumentSidebar: React.FC<VizDocumentSidebarProps> = ({
  chatId, // eslint-disable-line @typescript-eslint/no-unused-vars
  documentList,
  onDocumentUpload,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const files = e.dataTransfer?.files;
    if (files?.length) onDocumentUpload(files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files?.length) onDocumentUpload(files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const prettyDate = (iso: string) => {
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleDateString();
    } catch {
      return iso;
    }
  };

  const renderFileIcon = (type: VizDocument["type"]) => {
    switch (type) {
      case "pdf":
        return <FileText className="w-5 h-5 text-red-600 flex-shrink-0" aria-hidden="true" />;
      case "excel":
      case "csv":
        return <FileSpreadsheet className="w-5 h-5 text-green-600 flex-shrink-0" aria-hidden="true" />;
      case "folder":
        return <Folder className="w-5 h-5 text-blue-500 flex-shrink-0" aria-hidden="true" />;
      default:
        return <File className="w-5 h-5 text-gray-500 flex-shrink-0" aria-hidden="true" />;
    }
  };

  const renderStatusIcon = (status: VizDocument["status"]) => {
    switch (status) {
      case "uploaded":
        return <Clock className="w-4 h-4 text-yellow-500" aria-label="Uploaded" />;
      case "processing":
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" aria-label="Processing" />;
      case "ready":
        return <CheckCircle2 className="w-4 h-4 text-green-600" aria-label="Ready" />;
      default:
        return null;
    }
  };

  return (
    <aside className="w-80 bg-card border-r border-border flex flex-col">
      {/* Upload area */}
      <div className="p-4 border-b border-border">
        <div
          className={cn(
            "border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer",
            dragActive
              ? "border-accent bg-accent/10"
              : "border-muted-foreground/30 hover:border-accent hover:bg-accent/5"
          )}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          aria-label="Upload files"
        >
          <Upload className="w-8 h-8 mx-auto mb-3 text-muted-foreground" aria-hidden="true" />
          <p className="text-sm font-medium mb-2">
            Drop files here or{" "}
            <label className="text-accent cursor-pointer hover:underline">
              browse
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.xlsx,.xls,.csv"
                className="hidden"
                onChange={handleFileInput}
              />
            </label>
          </p>
          <p className="text-xs text-muted-foreground">PDF, Excel, or CSV files supported</p>
        </div>
      </div>

      {/* File list header */}
      <div className="p-4 border-b border-border">
        <h3 className="text-sm font-medium">
          Uploaded Files{" "}
          <span className="text-xs text-muted-foreground font-normal">
            ({documentList.length})
          </span>
        </h3>
      </div>

      {/* File list */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-2">
          {documentList.length === 0 ? (
            <p className="text-center text-muted-foreground text-sm py-8">
              No documents uploaded yet
            </p>
          ) : (
            documentList.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center space-x-3 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
                title={doc.name}
              >
                {renderFileIcon(doc.type)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{doc.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {doc.size} • {prettyDate(doc.uploadDate)}
                  </p>
                </div>
                {renderStatusIcon(doc.status)}
              </div>
            ))
          )}
        </div>
      </div>
    </aside>
  );
};
