// src/components/VizDocumentSidebar.tsx
import React, { useRef, useState } from "react";
import {
  Upload,
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
  chatId: string;
  documentList: VizDocument[];
  onDocumentUpload: (files: FileList) => Promise<void> | void;
}

const ALLOWED_EXT = /\.(xlsx|xls|csv)$/i;

const toFileList = (files: File[]): FileList => {
  const dt = new DataTransfer();
  files.forEach((f) => dt.items.add(f));
  return dt.files;
};

const filterExcelCsv = (list: FileList | null): File[] => {
  if (!list) return [];
  const out: File[] = [];
  for (const f of Array.from(list)) {
    if (ALLOWED_EXT.test(f.name)) out.push(f);
  }
  return out;
};

export const VizDocumentSidebar: React.FC<VizDocumentSidebarProps> = ({
  chatId, // eslint-disable-line @typescript-eslint/no-unused-vars
  documentList,
  onDocumentUpload,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const excelCsvInputRef = useRef<HTMLInputElement | null>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const accepted = filterExcelCsv(e.dataTransfer?.files || null);
    if (accepted.length) onDocumentUpload(toFileList(accepted));
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const accepted = filterExcelCsv(e.target.files);
    if (accepted.length) onDocumentUpload(toFileList(accepted));
    if (excelCsvInputRef.current) excelCsvInputRef.current.value = "";
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
            "border-2 border-dashed rounded-xl p-6 text-center transition-all",
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
          role="button"
          aria-label="Upload Excel or CSV"
          onClick={() => excelCsvInputRef.current?.click()}
        >
          <Upload className="w-8 h-8 mx-auto mb-3 text-muted-foreground" aria-hidden="true" />
          <h3 className="text-sm font-semibold mb-1"> Import your data</h3>
          <p className="text-xs text-muted-foreground mb-4">
            If you need some more data for visualization!
            {/* Supported formats: <strong>.xlsx</strong>, <strong>.xls</strong>, <strong>.csv</strong>. */}
          </p>

          <button
            type="button"
            onClick={() => excelCsvInputRef.current?.click()}
            className="inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
            aria-label="Upload Excel or CSV files"
          >
            <FileSpreadsheet className="w-4 h-4" />
            Upload
          </button>

          {/* Hidden input (Excel/CSV only) */}
          <input
            ref={excelCsvInputRef}
            type="file"
            multiple
            accept=".xlsx,.xls,.csv"
            className="hidden"
            onChange={handleFileInput}
          />
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
              No spreadsheets uploaded yet
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
