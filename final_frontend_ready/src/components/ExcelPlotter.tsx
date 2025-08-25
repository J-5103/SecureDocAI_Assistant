// src/components/ExcelPlotter.tsx
import React, { useState } from "react";
import { excelUpload, excelPlot } from "@/api/api";

export default function ExcelPlotter({ onSaved }: { onSaved?: (meta: any) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [filePath, setFilePath] = useState<string>("");
  const [question, setQuestion] = useState<string>("");
  const [title, setTitle] = useState<string>("");
  const [preview, setPreview] = useState<string>("");
  const [busy, setBusy] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setBusy(true);
    try {
      const res = await excelUpload(file);
      setFilePath(res.file_path);
    } catch (e: any) {
      alert(e.message || "Upload failed");
    } finally {
      setBusy(false);
    }
  };

  const handlePlot = async () => {
    if (!filePath || !question.trim()) return;
    setBusy(true);
    try {
      const res = await excelPlot(filePath, question, title || undefined);
      setPreview(`data:image/png;base64,${res.image_base64}`);
      onSaved?.(res.meta); // meta contains id, image_url, thumb_url etc.
    } catch (e: any) {
      alert(e.message || "Plot failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-3 p-4 border rounded-xl bg-card">
      <div className="grid md:grid-cols-2 gap-3">
        <input
          type="file"
          accept=".csv,.xls,.xlsx"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="border rounded p-2"
        />
        <button
          disabled={!file || busy}
          onClick={handleUpload}
          className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50"
        >
          {busy ? "Uploading..." : "Upload"}
        </button>
      </div>

      {filePath && (
        <div className="text-xs text-muted-foreground">Uploaded: {filePath}</div>
      )}

      <input
        placeholder='e.g. "bar plot of year and revenue"'
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        className="border rounded p-2"
      />
      <input
        placeholder="Title (optional)"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        className="border rounded p-2"
      />

      <button
        disabled={!filePath || !question.trim() || busy}
        onClick={handlePlot}
        className="px-4 py-2 rounded bg-green-600 text-white disabled:opacity-50"
      >
        {busy ? "Generating..." : "Generate Plot"}
      </button>

      {preview && (
        <div className="mt-3">
          <div className="font-medium mb-1">Preview</div>
          <img src={preview} alt="plot" className="w-full rounded border" />
        </div>
      )}
    </div>
  );
}
