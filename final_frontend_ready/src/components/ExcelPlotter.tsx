// src/components/ExcelPlotter.tsx
import React, { useEffect, useMemo, useState } from "react";
import { excelUpload, excelPlot, excelPlotCombine } from "@/api/api";

type RemoteFile = {
  name: string;
  file_path: string;
  size?: number;
  uploaded_at?: string;
};

const API_BASE =
  (import.meta as any)?.env?.VITE_API_BASE ||
  (process.env as any)?.REACT_APP_API_BASE ||
  "";

export default function ExcelPlotter({
  chatId: chatIdProp,
  onSaved,
}: {
  chatId?: string; // optional – if omitted, user can type one
  onSaved?: (meta: any) => void;
}) {
  // ---- identity / files ----------------------------------------------------
  const [chatId, setChatId] = useState<string>(chatIdProp || "");
  const [files, setFiles] = useState<RemoteFile[]>([]);
  const [loadingList, setLoadingList] = useState(false);

  const [combine, setCombine] = useState(false);
  const [selectedOne, setSelectedOne] = useState<string>(""); // file_path
  const [selectedMany, setSelectedMany] = useState<Record<string, boolean>>({});

  // ---- upload ---------------------------------------------------------------
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [busyUpload, setBusyUpload] = useState(false);

  // ---- question / output ----------------------------------------------------
  const [question, setQuestion] = useState("");
  const [title, setTitle] = useState("");
  const [busyGen, setBusyGen] = useState(false);
  const [preview, setPreview] = useState<string>(""); // data URL or absolute URL
  const [answer, setAnswer] = useState<string>(""); // numeric/stat fallback
  const [error, setError] = useState<string>("");

  // pickers
  const selectedPaths = useMemo(
    () => Object.keys(selectedMany).filter((p) => selectedMany[p]),
    [selectedMany]
  );

  // load list for chat
  const fetchList = async (cid: string) => {
    if (!cid) {
      setFiles([]);
      return;
    }
    setLoadingList(true);
    setError("");
    try {
      const res = await fetch(
        `${API_BASE}/api/excel/list?chat_id=${encodeURIComponent(cid)}`
      );
      if (!res.ok) {
        const msg = (await safeErr(res)) || "Failed to load files";
        throw new Error(msg);
      }
      const data = await res.json();
      const items: RemoteFile[] = (data?.files || []).map((f: any) => ({
        name: f.name,
        file_path: f.file_path,
        size: f.size,
        uploaded_at: f.uploaded_at,
      }));
      setFiles(items);
      // gently preselect latest if nothing chosen yet
      if (!combine && !selectedOne && items.length) {
        setSelectedOne(items[0].file_path);
      }
    } catch (e: any) {
      setError(e?.message || "Failed to load files");
      setFiles([]);
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    if (chatIdProp) setChatId(chatIdProp);
  }, [chatIdProp]);

  useEffect(() => {
    fetchList(chatId);
    // reset selections when chat changes
    setSelectedOne("");
    setSelectedMany({});
    setPreview("");
    setAnswer("");
    setError("");
  }, [chatId]);

  // ---- handlers -------------------------------------------------------------
  const handleUpload = async () => {
    if (!uploadFile) return;
    if (!chatId) {
      setError("Please enter/select a Chat ID before uploading.");
      return;
    }
    setBusyUpload(true);
    setError("");
    try {
      const res = await excelUpload(uploadFile, chatId); // your helper already supports chat_id
      const path = res?.file_path as string;
      // refresh list & select the newly uploaded file
      await fetchList(chatId);
      if (path) setSelectedOne(path);
    } catch (e: any) {
      setError(e?.message || "Upload failed");
    } finally {
      setBusyUpload(false);
    }
  };

  const handleGenerate = async () => {
    setBusyGen(true);
    setError("");
    setPreview("");
    setAnswer("");

    try {
      if (combine) {
        if (selectedPaths.length < 2) {
          throw new Error("Select at least two files to combine.");
        }
        const res = await excelPlotCombine(selectedPaths, question, title || undefined, chatId || undefined);
        // backend may return image_base64 or only image_url; and sometimes only `answer` for stats
        const b64 = res?.image_base64;
        const url = res?.image_url;
        const ans = res?.answer || res?.meta?.answer;
        if (b64) setPreview(`data:image/png;base64,${b64}`);
        else if (url) setPreview(withBase(url));
        if (ans) setAnswer(ans);
        onSaved?.(res?.meta || res);
      } else {
        const path = selectedOne;
        if (!path) throw new Error("Please select a file.");
        const res = await excelPlot(path, question, title || undefined, chatId || undefined);
        const b64 = res?.image_base64;
        const url = res?.image_url;
        const ans = res?.answer || res?.meta?.answer;
        if (b64) setPreview(`data:image/png;base64,${b64}`);
        else if (url) setPreview(withBase(url));
        if (ans) setAnswer(ans);
        onSaved?.(res?.meta || res);
      }
    } catch (e: any) {
      setError(e?.message || "Plot generation failed");
    } finally {
      setBusyGen(false);
    }
  };

  // ---- UI helpers -----------------------------------------------------------
  const toggleMany = (p: string) =>
    setSelectedMany((prev) => ({ ...prev, [p]: !prev[p] }));

  const withBase = (maybeRelative: string) => {
    if (!maybeRelative) return "";
    try {
      // If already absolute, return as is
      const u = new URL(maybeRelative, window.location.origin);
      if (/^https?:\/\//i.test(maybeRelative)) return maybeRelative;
      // otherwise prefix API_BASE (or stay relative if none)
      return API_BASE ? `${API_BASE}${maybeRelative}` : u.toString();
    } catch {
      return maybeRelative;
    }
  };

  return (
    <div className="grid gap-4 p-4 border rounded-xl bg-white">
      {/* Chat ID row (shown if not passed by parent) */}
      {!chatIdProp && (
        <div className="grid md:grid-cols-3 gap-3">
          <input
            placeholder="Chat ID (required to list files)"
            value={chatId}
            onChange={(e) => setChatId(e.target.value)}
            className="border rounded p-2"
          />
          <button
            onClick={() => fetchList(chatId)}
            className="px-3 py-2 rounded bg-gray-800 text-white disabled:opacity-50"
            disabled={!chatId || loadingList}
          >
            {loadingList ? "Loading..." : "Load Files"}
          </button>
          <div className="text-sm text-gray-500 self-center">
            Uploading or listing uses this chat's Excel folder.
          </div>
        </div>
      )}

      {/* Upload */}
      <div className="grid md:grid-cols-[1fr_auto] gap-3">
        <input
          type="file"
          accept=".csv,.xls,.xlsx"
          onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
          className="border rounded p-2"
        />
        <button
          disabled={!uploadFile || !chatId || busyUpload}
          onClick={handleUpload}
          className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50"
        >
          {busyUpload ? "Uploading..." : "Upload to Chat"}
        </button>
      </div>

      {/* Mode toggle */}
      <label className="inline-flex items-center gap-2">
        <input
          type="checkbox"
          checked={combine}
          onChange={(e) => {
            setCombine(e.target.checked);
            // clear outputs when switching mode
            setPreview("");
            setAnswer("");
            setError("");
          }}
        />
        <span className="text-sm">Combine multiple files</span>
      </label>

      {/* File pickers */}
      {!combine ? (
        <div className="grid gap-2">
          <div className="text-sm font-medium">Select file</div>
          <select
            value={selectedOne}
            onChange={(e) => setSelectedOne(e.target.value)}
            className="border rounded p-2"
          >
            <option value="">— choose —</option>
            {files.map((f) => (
              <option key={f.file_path} value={f.file_path}>
                {f.name}
              </option>
            ))}
          </select>
          {selectedOne && (
            <div className="text-xs text-gray-500 break-all">
              Using: {selectedOne}
            </div>
          )}
        </div>
      ) : (
        <div className="grid gap-2">
          <div className="text-sm font-medium">Select files to combine (2+)</div>
          <div className="max-h-48 overflow-auto border rounded p-2 space-y-1">
            {files.length === 0 && (
              <div className="text-sm text-gray-500">No files yet.</div>
            )}
            {files.map((f) => (
              <label key={f.file_path} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={!!selectedMany[f.file_path]}
                  onChange={() => toggleMany(f.file_path)}
                />
                <span className="text-sm">{f.name}</span>
              </label>
            ))}
          </div>
          <div className="text-xs text-gray-500">
            Selected: {selectedPaths.length}
          </div>
        </div>
      )}

      {/* Question + Title */}
      <input
        placeholder={`Ask for a specific plot, e.g. "bar chart of Sales by Region for 2019" or "line of revenue over date"`}
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
        disabled={
          busyGen ||
          !question.trim() ||
          (!combine ? !selectedOne : selectedPaths.length < 2)
        }
        onClick={handleGenerate}
        className="px-4 py-2 rounded bg-green-600 text-white disabled:opacity-50"
      >
        {busyGen ? "Generating..." : combine ? "Generate Combined Plot" : "Generate Plot"}
      </button>

      {/* Output */}
      {!!error && (
        <div className="p-3 rounded bg-red-50 text-red-700 text-sm">{error}</div>
      )}

      {!!answer && (
        <div className="p-3 rounded bg-amber-50 text-amber-800 text-sm">
          <div className="font-medium mb-1">Answer</div>
          <div>{answer}</div>
        </div>
      )}

      {!!preview && (
        <div className="mt-2">
          <div className="font-medium mb-1">Preview</div>
          <img src={preview} alt="plot" className="w-full rounded border" />
        </div>
      )}
    </div>
  );
}

// small util for error bodies
async function safeErr(res: Response) {
  try {
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const j = await res.json();
      return j?.detail || j?.error || "";
    }
    return await res.text();
  } catch {
    return "";
  }
}
