// src/components/NewVizChatModal.tsx
import React, { useState, useRef } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  onCreate: (chatName: string, files: FileList) => Promise<void> | void;
};

const allowed = [".csv", ".xls", ".xlsx"];

export default function NewVizChatModal({ open, onClose, onCreate }: Props) {
  const [chatName, setChatName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selected, setSelected] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const fileRef = useRef<HTMLInputElement | null>(null);

  if (!open) return null;

  const isAllowed = (name: string) =>
    allowed.some((ext) => name.toLowerCase().endsWith(ext));

  const validate = () => {
    if (!chatName.trim()) return "Please enter a chat name.";
    if (selected.length === 0)
      return "Please choose at least one Excel/CSV file.";
    const bad = selected.find((f) => !isAllowed(f.name));
    if (bad) return `Unsupported file: ${bad.name}. Only ${allowed.join(", ")}.`;
    return null;
  };

  const addFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    const newFiles = Array.from(files);
    const rejected = newFiles.filter((f) => !isAllowed(f.name));
    const allowedFiles = newFiles.filter((f) => isAllowed(f.name));

    setSelected((prev) => {
      // de-dupe by name + size (simple heuristic)
      const map = new Map<string, File>();
      [...prev, ...allowedFiles].forEach((f) =>
        map.set(`${f.name}-${f.size}`, f)
      );
      return Array.from(map.values());
    });

    if (rejected.length) {
      setError(
        `Skipped ${rejected.length} unsupported file(s). Allowed: ${allowed.join(
          ", "
        )}.`
      );
    } else {
      setError(null);
    }

    // reset native input value so the same file can be chosen again later
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    addFiles(e.target.files);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer?.files ?? null);
  };

  const removeFile = (idx: number) => {
    setSelected((prev) => prev.filter((_, i) => i !== idx));
    setError(null);
  };

  const clearFiles = () => {
    setSelected([]);
    setError(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleCreate = async () => {
    const msg = validate();
    if (msg) {
      setError(msg);
      return;
    }
    try {
      setSubmitting(true);
      // Build a FileList from our File[]
      const dt = new DataTransfer();
      selected.forEach((f) => dt.items.add(f));
      await onCreate(chatName.trim(), dt.files);
      // reset UI
      setChatName("");
      clearFiles();
    } finally {
      setSubmitting(false);
    }
  };

  const onKeyDownName = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      void handleCreate();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* Modal */}
      <div className="relative z-10 w-[640px] max-w-[92vw] rounded-xl bg-white p-6 shadow-xl">
        <h2 className="mb-4 text-2xl font-semibold">New Chat</h2>

        {/* Uploader (drag & drop + button) */}
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            Upload Document(s):
          </label>

          <div
            className={`border-2 border-dashed rounded-lg p-6 text-center transition-all ${
              dragOver
                ? "border-blue-500 bg-blue-50"
                : "border-gray-300 hover:border-blue-400 hover:bg-blue-50/40"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            role="button"
            aria-label="Upload files"
          >
            <p className="text-sm">
              Drag & drop files here, or{" "}
              <span className="text-blue-600 underline">browse</span>
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Allowed: .csv, .xls, .xlsx (multiple files supported)
            </p>
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".csv,.xls,.xlsx"
              className="hidden"
              onChange={handleInputChange}
            />
          </div>
        </div>

        {/* Selected files list */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">
              Selected Files{" "}
              <span className="text-xs text-gray-500">
                ({selected.length})
              </span>
            </span>
            {selected.length > 0 && (
              <button
                onClick={clearFiles}
                className="text-xs text-gray-600 hover:text-gray-900"
              >
                Clear all
              </button>
            )}
          </div>

          {selected.length === 0 ? (
            <p className="text-xs text-gray-500">No files selected yet.</p>
          ) : (
            <ul className="max-h-40 overflow-y-auto divide-y rounded border">
              {selected.map((f, idx) => (
                <li
                  key={`${f.name}-${f.size}-${idx}`}
                  className="flex items-center justify-between px-3 py-2 text-sm"
                >
                  <div className="truncate">
                    <span className="font-medium">{f.name}</span>
                    <span className="ml-2 text-gray-500">
                      {(f.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                  <button
                    onClick={() => removeFile(idx)}
                    className="ml-3 rounded px-2 py-1 text-xs text-red-600 hover:bg-red-50"
                    aria-label={`Remove ${f.name}`}
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Chat name */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Chat Name:</label>
          <input
            value={chatName}
            onChange={(e) => setChatName(e.target.value)}
            onKeyDown={onKeyDownName}
            className="block w-full rounded border px-3 py-2"
            placeholder="e.g. Sales Q2"
          />
        </div>

        {/* Errors */}
        {error && (
          <div className="mb-4 rounded-md bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-3">
          <button
            onClick={onClose}
            className="rounded-lg border px-4 py-2 hover:bg-gray-50"
            disabled={submitting}
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={submitting}
            className={`rounded-lg bg-blue-600 px-5 py-2 text-white hover:bg-blue-700 disabled:opacity-60 ${
              submitting ? "cursor-wait" : ""
            }`}
          >
            {submitting ? "Creating..." : "Create"}
          </button>
        </div>
      </div>
    </div>
  );
}
