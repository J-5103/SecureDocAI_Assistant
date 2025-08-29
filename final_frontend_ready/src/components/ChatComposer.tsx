import React, { useEffect, useMemo, useRef, useState } from "react";

type DocItem = {
  name: string;           // filename with extension
  documentId: string;     // filename with extension (as backend returns in list)
  size: number;
  ready?: boolean;        // becomes true when vectorstore is ready
  taskId?: string;        // for polling
  kind: "pdf" | "doc" | "sheet" | "image" | "other";
  localPreviewUrl?: string; // object URL for images (lives until page refresh)
};

type UserMsg = {
  role: "user";
  text: string;
  attachments?: { name: string; kind: DocItem["kind"]; previewUrl?: string }[];
};

type AssistantMsg = { role: "assistant"; text: string };

type ChatMsg = UserMsg | AssistantMsg;

const API_BASE =
  (typeof import.meta !== "undefined" &&
    (import.meta as any).env &&
    (import.meta as any).env.VITE_API_BASE) ||
  (window as any).__API_BASE__ ||
  "http://192.168.0.110:8000";

// ---------- utils ----------
const extOf = (n?: string) => (n ? n.split(".").pop()?.toLowerCase() || "" : "");
const isImg = (n?: string) => ["png", "jpg", "jpeg"].includes(extOf(n));
const isSheet = (n?: string) => ["xls", "xlsx", "csv"].includes(extOf(n));
const isPdf = (n?: string) => extOf(n) === "pdf";
const isDoc = (n?: string) => ["doc", "docx"].includes(extOf(n));

const kindOf = (n: string): DocItem["kind"] =>
  isImg(n) ? "image" : isSheet(n) ? "sheet" : isPdf(n) ? "pdf" : isDoc(n) ? "doc" : "other";

const iconOf = (k: DocItem["kind"]) =>
  k === "image" ? "🖼️" : k === "pdf" ? "📄" : k === "sheet" ? "📊" : k === "doc" ? "📝" : "📎";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function useStickyChatId(key = "doc_chat_id") {
  const [chatId, setChatId] = useState<string | null>(null);
  useEffect(() => {
    let cid = localStorage.getItem(key);
    if (!cid) {
      // simple 8-char id
      cid = `chat-${Math.random().toString(16).slice(2, 10)}`;
      localStorage.setItem(key, cid);
    }
    setChatId(cid);
  }, [key]);
  return [chatId, setChatId] as const;
}

// ---------- main component ----------
export default function ChatComposer() {
  const [chatId] = useStickyChatId();
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);

  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [error, setError] = useState<string | null>(null);

  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);

  const fileRef = useRef<HTMLInputElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  // autoscroll chat
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // fetch doc list on load / chat change
  useEffect(() => {
    if (!chatId) return;
    refreshDocs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatId]);

  // helpful chips: show previews for image docs we have local preview for (most recent first)
  const chipImages = useMemo(
    () =>
      docs
        .filter((d) => d.kind === "image" && d.localPreviewUrl)
        .slice(-4)
        .map((d) => d.localPreviewUrl!) // non-null by filter
        .reverse(),
    [docs]
  );

  async function refreshDocs() {
    if (!chatId) return;
    try {
      const r = await fetch(`${API_BASE}/api/list_documents?chat_id=${encodeURIComponent(chatId)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const items = Array.isArray(data?.documents) ? data.documents : [];
      setDocs((old) => {
        const prevPreviewByName = new Map(old.map((d) => [d.name, d.localPreviewUrl]));
        return items.map((it: any) => ({
          name: it.name,
          documentId: it.documentId ?? it.name,
          size: it.size ?? 0,
          ready: true, // when listed, assume processed or soon to be
          kind: kindOf(it.name),
          localPreviewUrl: prevPreviewByName.get(it.name),
        }));
      });
    } catch {
      // ignore errors here
    }
  }

  async function pollTask(taskId: string, onUpdate: (status: any) => void) {
    // poll up to ~90s
    for (let i = 0; i < 45; i++) {
      try {
        const r = await fetch(`${API_BASE}/api/status/${taskId}`);
        if (r.ok) {
          const data = await r.json();
          onUpdate(data);
          if (data?.status === "ready" || data?.status === "failed") return data;
        }
      } catch {
        // ignore
      }
      await sleep(2000);
    }
    return { status: "timeout" };
  }

  async function handleUploadFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    if (!chatId) return setError("No chat id yet.");

    setError(null);
    setUploading(true);

    // Build a local preview for images (will live until refresh)
    let previewUrl: string | undefined;
    if (isImg(f.name)) previewUrl = URL.createObjectURL(f);

    try {
      const fd = new FormData();
      fd.append("chat_id", chatId);
      fd.append("file", f);

      const r = await fetch(`${API_BASE}/api/upload/upload_file`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`Upload failed: HTTP ${r.status}`);
      const data = await r.json();

      // Pre-append to the docs list as "processing"
      const optimistic: DocItem = {
        name: data.filename || f.name,
        documentId: (data.document_id && data.filename) ? `${data.document_id}${extOf(data.filename) ? "" : ""}` : (f.name),
        size: f.size,
        ready: (data.status === "ready"),
        taskId: data.task_id,
        kind: kindOf(f.name),
        localPreviewUrl: previewUrl,
      };
      setDocs((d) => [...d, optimistic]);

      // Poll background vectorstore build
      if (data.task_id && data.status === "processing") {
        await pollTask(data.task_id, (st) => {
          setDocs((dd) =>
            dd.map((x) =>
              x.taskId === data.task_id
                ? { ...x, ready: st?.status === "ready" ? true : x.ready }
                : x
            )
          );
        });
        // refresh list once ready
        await refreshDocs();
      } else {
        // if sync/ready, just refresh list
        await refreshDocs();
      }

      // auto-select just uploaded doc
      setSelectedDocId(f.name);
    } catch (err: any) {
      setError(err.message || "Upload failed.");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function sendQuestion() {
    if (!chatId) return setError("No chat id yet.");
    const q = input.trim();
    if (!q) return;

    setError(null);
    setSending(true);

    // Resolve selected doc (if any) & find preview for thumbnails in the user bubble
    const selected = docs.find((d) => d.documentId === selectedDocId || d.name === selectedDocId);
    const attachments =
      selected && selected.kind === "image"
        ? [{ name: selected.name, kind: selected.kind, previewUrl: selected.localPreviewUrl }]
        : undefined;

    // Push user message first
    setMessages((m) => [...m, { role: "user", text: q, attachments }]);
    setInput("");

    try {
      const body: any = {
        chat_id: chatId,
        question: q,
      };
      if (selected?.documentId) body.document_id = selected.documentId;

      const r = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();

      const answer = typeof data?.answer === "string" ? data.answer : JSON.stringify(data, null, 2);
      setMessages((m) => [...m, { role: "assistant", text: answer }]);
    } catch (err: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: `❌ Failed to get answer: ${err.message || err}` },
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div style={styles.wrap}>
      <h2>📚 Document Q&A</h2>

      {/* Upload + doc list */}
      <div style={styles.panel}>
        <div style={styles.uploadRow}>
          <input
            ref={fileRef}
            type="file"
            onChange={handleUploadFile}
            accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.png,.jpg,.jpeg"
            disabled={uploading}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            style={styles.smallBtn}
            disabled={uploading}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </div>

        {/* tiny image chips (latest uploads with local preview) */}
        {chipImages.length > 0 && (
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            {chipImages.map((u, i) => (
              <div key={i} style={{ width: 48, height: 48 }}>
                <img
                  src={u}
                  alt=""
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "cover",
                    borderRadius: 8,
                    border: "1px solid #ddd",
                  }}
                />
              </div>
            ))}
          </div>
        )}

        <div style={styles.docList}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Your Files</div>
          {docs.length === 0 && <div style={{ opacity: 0.7 }}>No files yet.</div>}
          {docs.map((d) => {
            const sel = d.documentId === selectedDocId || d.name === selectedDocId;
            return (
              <div
                key={d.name}
                onClick={() => setSelectedDocId(d.documentId)}
                style={{
                  ...styles.docItem,
                  borderColor: sel ? "#1565c0" : "#e0e0e0",
                  background: sel ? "#e3f2fd" : "#fff",
                  cursor: "pointer",
                }}
                title={d.name}
              >
                <div style={{ fontSize: 18 }}>{iconOf(d.kind)}</div>
                <div style={{ marginLeft: 8, flex: 1, minWidth: 0 }}>
                  <div style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                    {d.name}
                  </div>
                  <div style={{ fontSize: 12, opacity: 0.7 }}>
                    {d.ready ? "Ready" : "Processing…"} • {(d.size / 1024).toFixed(1)} KB
                  </div>
                </div>
                {d.kind === "image" && d.localPreviewUrl && (
                  <img
                    src={d.localPreviewUrl}
                    alt=""
                    style={{ width: 48, height: 48, objectFit: "cover", borderRadius: 6 }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Chat area */}
      <div style={styles.chatBox}>
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} style={{ textAlign: "right", margin: "10px 0" }}>
              <div style={{ display: "inline-block", background: "#e3f2fd", padding: 10, borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>You</div>
                <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>
                {m.attachments && m.attachments.length > 0 && (
                  <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                    {m.attachments.map((a, idx) => (
                      <div key={idx} title={a.name}>
                        {a.previewUrl ? (
                          <img
                            src={a.previewUrl}
                            alt=""
                            style={{ width: 80, height: 80, objectFit: "cover", borderRadius: 6 }}
                          />
                        ) : (
                          <div
                            style={{
                              width: 80,
                              height: 80,
                              borderRadius: 6,
                              border: "1px solid #ddd",
                              display: "grid",
                              placeItems: "center",
                              fontSize: 20,
                            }}
                          >
                            {iconOf(a.kind)}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div key={i} style={{ textAlign: "left", margin: "10px 0" }}>
              <div style={{ display: "inline-block", background: "#f1f8e9", padding: 10, borderRadius: 8, maxWidth: 760 }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>Assistant</div>
                <div style={{ whiteSpace: "pre-wrap" }}>{m.text}</div>
              </div>
            </div>
          )
        )}
        <div ref={endRef} />
      </div>

      {/* Composer */}
      <div style={styles.composer}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            selectedDocId
              ? `Ask about "${selectedDocId}"…`
              : "Ask a question (select a file to focus on it)…"
          }
          style={styles.textarea}
          rows={3}
        />
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={sendQuestion} style={styles.sendBtn} disabled={sending || !input.trim()}>
            {sending ? "Asking…" : "Ask"}
          </button>
          {selectedDocId && (
            <span style={{ fontSize: 12, opacity: 0.75 }}>
              Focus: <strong>{selectedDocId}</strong>
            </span>
          )}
        </div>
      </div>

      {error && <div style={{ color: "crimson", marginTop: 10 }}>{error}</div>}
    </div>
  );
}

// ---------- styles ----------
const styles: Record<string, React.CSSProperties> = {
  wrap: {
    maxWidth: 920,
    margin: "30px auto",
    padding: "16px",
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
  },
  panel: {
    border: "1px solid #e0e0e0",
    borderRadius: 10,
    padding: 12,
    background: "#fff",
  },
  uploadRow: { display: "flex", gap: 10, alignItems: "center" },
  smallBtn: {
    padding: "6px 10px",
    background: "#263238",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
  },
  docList: { marginTop: 12, display: "grid", gap: 8 },
  docItem: {
    display: "flex",
    alignItems: "center",
    padding: 8,
    borderRadius: 8,
    border: "1px solid #e0e0e0",
  },
  chatBox: {
    marginTop: 16,
    border: "1px solid #e0e0e0",
    borderRadius: 10,
    padding: 12,
    minHeight: 220,
    background: "#fff",
  },
  composer: {
    marginTop: 12,
    border: "1px solid #e0e0e0",
    borderRadius: 10,
    padding: 10,
    background: "#fff",
  },
  textarea: {
    width: "100%",
    resize: "vertical",
    border: "1px solid #cfd8dc",
    borderRadius: 8,
    padding: 10,
    outline: "none",
    fontSize: 14,
    marginBottom: 8,
  },
  sendBtn: {
    padding: "8px 14px",
    background: "#1565c0",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
  },
};
