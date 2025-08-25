import React, { useEffect, useRef, useState } from "react";

type UserImageMsg = {
  role: "user";
  type: "image";
  frontName: string;
  backName?: string;
  urls: string[];
};

type AssistantMsg = {
  role: "assistant";
  whatsapp: string;
  vcard: string;
  json: any;
};

type Msg = UserImageMsg | AssistantMsg;

/** Backend base:
 * - If VITE_API_BASE is set (e.g. http://192.168.0.109:8000), we use it.
 * - Else empty string → relative paths (/api, /static) that Vite proxies in dev.
 */
const RAW_API_BASE =
  (typeof import.meta !== "undefined" &&
    (import.meta as any).env &&
    (import.meta as any).env.VITE_API_BASE) ||
  (window as any).__API_BASE__ ||
  "";

// normalize (no trailing slash)
const API_BASE = String(RAW_API_BASE).replace(/\/+$/, "");

/** Resolve a server-provided path. If it’s relative like "/static/...",
 * prefix with API_BASE so images still load when backend is on another host.
 * Blob/object URLs and absolute URLs are returned unchanged.
 */
function resolveUrl(u: string): string {
  if (!u) return u;
  if (u.startsWith("blob:") || u.startsWith("data:")) return u;
  if (/^https?:\/\//i.test(u)) return u;
  if (u.startsWith("/")) return `${API_BASE}${u}`;
  return u;
}

function ImageCardExtractor() {
  const [frontImage, setFrontImage] = useState<File | null>(null);
  const [backImage, setBackImage] = useState<File | null>(null);

  // tiny chip previews above the input
  const [chipUrls, setChipUrls] = useState<string[]>([]);

  // last extracted payload (for quick display + vCard download)
  const [whatsapp, setWhatsapp] = useState("");
  const [vcard, setVcard] = useState("");
  const [jsonOut, setJsonOut] = useState<any>(null);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileFrontRef = useRef<HTMLInputElement | null>(null);
  const fileBackRef = useRef<HTMLInputElement | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Build tiny chip previews when user selects images (auto-cleanup)
  useEffect(() => {
    const urls: string[] = [];
    if (frontImage) urls.push(URL.createObjectURL(frontImage));
    if (backImage) urls.push(URL.createObjectURL(backImage));
    setChipUrls(urls);

    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [frontImage, backImage]);

  // Auto scroll to bottom when messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Restore session from localStorage and fetch chat history
  useEffect(() => {
    const saved = localStorage.getItem("vision_session_id");
    if (saved) {
      setSessionId(saved);
    }
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    localStorage.setItem("vision_session_id", sessionId);
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/api/chat/${sessionId}`);
        if (!r.ok) return;
        const data = await r.json();
        const history = Array.isArray(data?.history) ? data.history : [];
        const mapped: Msg[] = history
          .map((h: any) => {
            if (h?.role === "user" && h?.type === "image") {
              const rawUrls = Array.isArray(h.image_urls) ? h.image_urls : [];
              const urls = rawUrls.map((u: string) => resolveUrl(u));
              return {
                role: "user",
                type: "image",
                frontName: h.front_name || "front",
                backName: h.back_name || undefined,
                urls,
              };
            }
            if (h?.role === "assistant") {
              return {
                role: "assistant",
                whatsapp: h.whatsapp || "",
                vcard: h.vcard || "",
                json: h.json ?? {},
              };
            }
            return null;
          })
          .filter(Boolean) as Msg[];
        setMessages(mapped);
      } catch {
        /* ignore */
      }
    })();
  }, [sessionId]);

  const handleSubmit = async () => {
    if (!frontImage) {
      setError("Please select the front image.");
      return;
    }
    setError(null);
    setWhatsapp("");
    setVcard("");
    setJsonOut(null);
    setLoading(true);

    // create local preview URLs for the chat bubble (separate from chipUrls)
    const localUrls: string[] = [
      URL.createObjectURL(frontImage),
      ...(backImage ? [URL.createObjectURL(backImage)] : []),
    ];

    // push the user bubble immediately
    setMessages((m) => [
      ...m,
      {
        role: "user",
        type: "image",
        frontName: frontImage.name,
        backName: backImage?.name,
        urls: localUrls,
      },
    ]);

    try {
      const formData = new FormData();
      formData.append("front_image", frontImage);
      if (backImage) formData.append("back_image", backImage);
      if (sessionId) formData.append("session_id", sessionId);

      const res = await fetch(`${API_BASE}/api/ask-image`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.status !== "success") {
        throw new Error(data.message || "Failed to extract.");
      }

      // replace the last user bubble's local URLs with server URLs (persistent)
      const serverUrlsRaw: string[] = Array.isArray(data?.data?.image_urls)
        ? data.data.image_urls
        : [];
      const serverUrls = serverUrlsRaw.map((u) => resolveUrl(u));

      setMessages((m) => {
        const cloned = [...m];
        // last message is the user bubble we just pushed
        const idx = cloned.length - 1;
        const last = cloned[idx] as Msg | undefined;
        if (last && last.role === "user" && "urls" in last) {
          cloned[idx] = { ...last, urls: serverUrls.length ? serverUrls : last.urls };
        }
        return cloned;
      });

      // now push assistant bubble
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          whatsapp: data.data.whatsapp || "",
          vcard: data.data.vcard || "",
          json: data.data.json ?? {},
        },
      ]);

      setWhatsapp(data.data.whatsapp || "");
      setVcard(data.data.vcard || "");
      setJsonOut(data.data.json ?? null);
      if (data.session_id) setSessionId(data.session_id);

      // clear selections (optional)
      setFrontImage(null);
      setBackImage(null);
      if (fileFrontRef.current) fileFrontRef.current.value = "";
      if (fileBackRef.current) fileBackRef.current.value = "";
    } catch (err: any) {
      setError(err.message || "Error uploading image or connecting to server.");
    } finally {
      // cleanup local ObjectURLs used for the chat bubble (after we swapped to server URLs)
      setTimeout(() => localUrls.forEach((u) => URL.revokeObjectURL(u)), 1000);
      setLoading(false);
    }
  };

  const downloadVcf = () => {
    if (!vcard) return;
    const blob = new Blob([vcard], { type: "text/vcard;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "contact.vcf";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={styles.container}>
      <h2>🖼️ Business Card Extractor</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <label style={styles.label}>Front Image (required):</label>
          <input
            ref={fileFrontRef}
            type="file"
            accept="image/*"
            onChange={(e) => setFrontImage(e.target.files?.[0] || null)}
          />
        </div>
        <div>
          <label style={styles.label}>Back Image (optional):</label>
          <input
            ref={fileBackRef}
            type="file"
            accept="image/*"
            onChange={(e) => setBackImage(e.target.files?.[0] || null)}
          />
        </div>
      </div>

      {/* tiny thumbnail chips near the composer */}
      {chipUrls.length > 0 && (
        <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
          {chipUrls.map((u, i) => (
            <div key={i} style={{ position: "relative", width: 56, height: 56 }}>
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

      <button onClick={handleSubmit} style={styles.button} disabled={loading}>
        {loading ? "Processing..." : "Extract Info"}
      </button>

      {error && <p style={styles.error}>{error}</p>}

      {/* Chat-style visualization */}
      <div style={{ marginTop: 16 }}>
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} style={{ margin: "12px 0", textAlign: "right" }}>
              <div
                style={{
                  display: "inline-block",
                  background: "#e3f2fd",
                  padding: 10,
                  borderRadius: 8,
                  maxWidth: 620,
                }}
              >
                <div style={{ fontWeight: 600 }}>🧑‍💻 You uploaded:</div>
                {msg.urls?.length > 0 && (
                  <div style={{ display: "flex", gap: 8, marginTop: 6, flexWrap: "wrap" }}>
                    {msg.urls.map((u, idx) => (
                      <a href={u} target="_blank" rel="noreferrer" key={idx} title="Open image">
                        <img
                          src={u}
                          alt=""
                          style={{ width: 120, height: 120, objectFit: "cover", borderRadius: 6 }}
                        />
                      </a>
                    ))}
                  </div>
                )}
                <div style={{ opacity: 0.7, marginTop: 6 }}>
                  {msg.frontName}
                  {msg.backName ? ` + ${msg.backName}` : ""}
                </div>
              </div>
            </div>
          ) : (
            <div key={i} style={{ margin: "12px 0", textAlign: "left" }}>
              <div
                style={{
                  display: "inline-block",
                  background: "#f1f8e9",
                  padding: 10,
                  borderRadius: 8,
                  maxWidth: 640,
                }}
              >
                <div style={{ fontWeight: 700 }}>🤖 Extracted</div>
                <div style={{ whiteSpace: "pre-wrap", marginTop: 6 }}>{msg.whatsapp}</div>
                <details style={{ marginTop: 8 }}>
                  <summary>vCard</summary>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{msg.vcard}</pre>
                </details>
                <details style={{ marginTop: 8 }}>
                  <summary>JSON</summary>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(msg.json, null, 2)}</pre>
                </details>
                <button
                  onClick={downloadVcf}
                  style={{ ...styles.button, backgroundColor: "#1b5e20", marginTop: 8 }}
                >
                  Download vCard
                </button>
              </div>
            </div>
          )
        )}
        <div ref={chatEndRef} />
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "700px",
    margin: "40px auto",
    padding: "20px",
    border: "1px solid #ccc",
    borderRadius: "8px",
    backgroundColor: "#f9f9f9",
    fontFamily: "sans-serif",
  } as React.CSSProperties,
  label: { display: "block", marginTop: "12px", fontWeight: "bold" } as React.CSSProperties,
  button: {
    marginTop: "20px",
    padding: "10px 20px",
    backgroundColor: "#004d40",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  } as React.CSSProperties,
  error: { color: "red", marginTop: "12px" } as React.CSSProperties,
};

export default ImageCardExtractor;
