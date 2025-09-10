// src/components/ChatInterface.tsx
import { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { Send, Bot, User, Sparkles, X, Link as LinkIcon, ImagePlus, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import ReactMarkdown from "react-markdown";

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: string;
  /** single image (legacy) */
  imageUrl?: string;
  /** Some API paths return plotImageUrl instead of imageUrl */
  plotImageUrl?: string;
  /** NEW: support multiple images in one message */
  imageUrls?: string[];
  imageAlt?: string;
}

interface Document {
  documentId: string;
  name: string;
}

interface ChatInterfaceProps {
  messages?: Message[];
  /** onSendMessage(question, documentId?, docIds?, images?) */
  onSendMessage: (
    question: string,
    documentId?: string | null,
    docIds?: string[],
    images?: File[]
  ) => void | Promise<void>;
  documents?: Document[];
  isLoading?: boolean;

  /** NEW: current chat id for per-chat card export & for parent to tag uploads */
  chatId?: string;
  /** NEW: optional pretty title for the export file name */
  chatTitle?: string;
}

/* -------------------- Image reduction helpers -------------------- */
type ReduceOpts = {
  maxW?: number;
  maxH?: number;
  maxBytes?: number;
  mimeType?: "image/webp" | "image/jpeg";
  stepQuality?: number;
  minQuality?: number;
  initialQuality?: number;
};

const defaultReduceOpts: Required<ReduceOpts> = {
  maxW: 1600,
  maxH: 1600,
  maxBytes: 800_000, // ~0.8MB
  mimeType: "image/webp",
  stepQuality: 0.07,
  minQuality: 0.5,
  initialQuality: 0.92,
};

function loadImageFromFile(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };
    img.src = url;
  });
}

function drawToCanvas(img: HTMLImageElement, maxW: number, maxH: number): HTMLCanvasElement {
  let w = img.width;
  let h = img.height;
  const scale = Math.min(maxW / w, maxH / h, 1); // only downscale
  w = Math.floor(w * scale);
  h = Math.floor(h * scale);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context not available");
  ctx.drawImage(img, 0, 0, w, h);
  return canvas;
}

async function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Canvas toBlob failed"))),
      type,
      quality
    );
  });
}

async function reduceImageFile(file: File, opts: ReduceOpts = {}): Promise<File> {
  const { maxW, maxH, maxBytes, mimeType, stepQuality, minQuality, initialQuality } = {
    ...defaultReduceOpts,
    ...opts,
  };

  // already small enough — pass through
  if (file.size <= maxBytes && file.type.startsWith("image/")) return file;

  const img = await loadImageFromFile(file);
  const canvas = drawToCanvas(img, maxW, maxH);

  let q = initialQuality;
  let blob = await canvasToBlob(canvas, mimeType, q);

  while (blob.size > maxBytes && q > minQuality) {
    q = Math.max(minQuality, q - stepQuality);
    blob = await canvasToBlob(canvas, mimeType, q);
  }

  const ext = mimeType === "image/webp" ? "webp" : "jpg";
  const base = file.name.replace(/\.[^.]+$/, "");
  const reducedName = `${base}-reduced.${ext}`;
  return new File([blob], reducedName, { type: mimeType, lastModified: Date.now() });
}

const toObjectUrl = (f: Blob) => URL.createObjectURL(f);
/* ----------------------------------------------------------------- */

/** robust detector so we don't show a duplicate "Thinking…" bubble */
const isThinkingMsg = (m?: Message) => {
  if (!m) return false;
  const lettersOnly = (m.text || "").trim().toLowerCase().replace(/[^a-z]/g, "");
  return lettersOnly === "thinking";
};

export const ChatInterface = ({
  messages = [],
  onSendMessage,
  documents = [],
  isLoading = false,
  chatId,       // NEW
  chatTitle,    // NEW
}: ChatInterfaceProps) => {
  const [inputText, setInputText] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string>("");
  const [isCombineMode, setIsCombineMode] = useState(false);
  const [combineDialogOpen, setCombineDialogOpen] = useState(false);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  // attachments
  const [images, setImages] = useState<File[]>([]);      // includes images AND PDFs for sending
  const [previews, setPreviews] = useState<string[]>([]); // thumbnails for images only
  const [pdfNames, setPdfNames] = useState<string[]>([]); // chips for PDFs
  const imgInputRef = useRef<HTMLInputElement>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const composingRef = useRef(false);

  // drag & drop highlight
  const [isDragging, setIsDragging] = useState(false);

  // NEW: export loading state
  const [downloading, setDownloading] = useState(false);

  // ---- memo helpers ----
  const nameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const d of documents) m.set(d.documentId, d.name);
    return m;
  }, [documents]);

  const chipNames = useMemo(() => {
    if (isCombineMode) return selectedDocIds.map((id) => nameById.get(id) || id);
    if (selectedDocId) return [nameById.get(selectedDocId) || selectedDocId];
    return [];
  }, [isCombineMode, selectedDocIds, selectedDocId, nameById]);

  // ---- effects ----
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // keep selection valid
  useEffect(() => {
    if (!documents || documents.length === 0) {
      if (selectedDocId !== "" || isCombineMode || selectedDocIds.length) {
        setSelectedDocId("");
        setIsCombineMode(false);
        setSelectedDocIds([]);
      }
      return;
    }
    const hasSelected = !!selectedDocId && documents.some((d) => d.documentId === selectedDocId);
    if (!isCombineMode && !hasSelected) {
      const first = documents[0].documentId;
      if (selectedDocId !== first) setSelectedDocId(first);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents]);

  // cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      previews.forEach((u) => URL.revokeObjectURL(u));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- file type helpers (match Viz behavior) ----
  const looksLikeImage = useCallback(
    (f: File) =>
      (f.type && f.type.startsWith("image/")) ||
      /\.(png|jpe?g|webp|gif|bmp|tiff)$/i.test(f.name || ""),
    []
  );
  const looksLikePdf = useCallback(
    (f: File) =>
      (f.type && f.type.toLowerCase() === "application/pdf") || /\.pdf$/i.test(f.name || ""),
    []
  );

  // ---- add attachments (reduce images; PDFs pass-through) ----
  const addImages = useCallback(
    async (files: FileList | File[] | null) => {
      if (!files) return;
      const arr = Array.from(files);

      const toSend: File[] = [];
      const newPdfNames: string[] = [];
      const imgFiles: File[] = [];

      for (const f of arr) {
        if (looksLikeImage(f)) {
          try {
            const small = await reduceImageFile(f);
            toSend.push(small);
            imgFiles.push(small);
          } catch {
            toSend.push(f);
            imgFiles.push(f);
          }
        } else if (looksLikePdf(f)) {
          toSend.push(f);
          newPdfNames.push(f.name);
        }
      }

      if (!toSend.length) return;

      const urls = await Promise.all(
        imgFiles.map(async (f) => {
          try {
            return toObjectUrl(f);
          } catch {
            return "";
          }
        })
      );

      setImages((p) => [...p, ...toSend]);
      setPreviews((p) => [...p, ...urls.filter(Boolean)]);
      setPdfNames((p) => [...p, ...newPdfNames]);

      if (imgInputRef.current) imgInputRef.current.value = "";
    },
    [looksLikeImage, looksLikePdf]
  );

  const removePreview = useCallback((idx: number) => {
    setPreviews((prev) => {
      const url = prev[idx];
      if (url) URL.revokeObjectURL(url);
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
    setImages((prev) => {
      // remove the Nth image file (ignore PDFs)
      const imageIndexes = prev
        .map((f, i) => ({ f, i }))
        .filter(({ f }) => looksLikeImage(f))
        .map(({ i }) => i);
      const underlyingIndex = imageIndexes[idx];
      if (underlyingIndex == null) return prev;
      const copy = [...prev];
      copy.splice(underlyingIndex, 1);
      return copy;
    });
  }, [looksLikeImage]);

  const removePdf = useCallback((idx: number) => {
    const name = pdfNames[idx];
    setPdfNames((prev) => {
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
    setImages((prev) => {
      const pos = prev.findIndex((f) => looksLikePdf(f) && f.name === name);
      if (pos < 0) return prev;
      const copy = [...prev];
      copy.splice(pos, 1);
      return copy;
    });
  }, [pdfNames, looksLikePdf]);

  const clearImages = useCallback(() => {
    previews.forEach((u) => URL.revokeObjectURL(u));
    setPreviews([]);
    setPdfNames([]);
    setImages([]);
    if (imgInputRef.current) imgInputRef.current.value = "";
  }, [previews]);

  // Paste images from clipboard
  const handlePaste = useCallback(
    async (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      const files: File[] = [];
      for (const it of items as any) {
        if (it.kind === "file") {
          const f = it.getAsFile?.();
          if (f && f.type.startsWith("image/")) files.push(f);
        }
      }
      if (files.length) {
        e.preventDefault();
        await addImages(files);
      }
    },
    [addImages]
  );

  // Drag & drop images/PDFs
  const handleDrop = useCallback(
    async (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      if (isLoading) return;
      const files = e.dataTransfer?.files || null;
      if (files && files.length) {
        await addImages(files);
      }
    },
    [addImages, isLoading]
  );
  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);
  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);
  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  // ---- submit ----
  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = inputText.trim();
      if ((!trimmed && images.length === 0) || isLoading) return;

      // NOTE: your parent should include `chatId` when sending images for card extraction
      if (isCombineMode && selectedDocIds.length > 0) {
        await onSendMessage(trimmed, null, selectedDocIds, images);
      } else if (!isCombineMode && selectedDocId) {
        await onSendMessage(trimmed, selectedDocId, undefined, images);
      } else if (images.length > 0) {
        // allow pure vision questions with no doc context
        await onSendMessage(trimmed, null, undefined, images);
      } else {
        return;
      }

      setInputText("");
      clearImages();
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    },
    [inputText, images, isLoading, isCombineMode, selectedDocIds, selectedDocId, onSendMessage, clearImages]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (composingRef.current) return;
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void handleSubmit(e as unknown as React.FormEvent);
      }
    },
    [handleSubmit]
  );

  const handleTextareaChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  }, []);

  const handleDropdownChange = useCallback((value: string) => {
    if (value === "combine") {
      // Preselect current single doc (if any) to make it faster
      setSelectedDocIds((prev) => (prev.length > 0 ? prev : selectedDocId ? [selectedDocId] : []));
      setCombineDialogOpen(true);
    } else {
      setIsCombineMode(false);
      setSelectedDocIds([]);
      setSelectedDocId(value);
    }
  }, [selectedDocId]);

  const toggleCombineDoc = useCallback(
    (docId: string, checked?: boolean | string) => {
      const willSelect =
        checked === true || (checked === undefined && !selectedDocIds.includes(docId));
      setSelectedDocIds((prev) => (willSelect ? [...prev, docId] : prev.filter((id) => id !== docId)));
    },
    [selectedDocIds]
  );

  const handleCombineConfirm = useCallback(() => {
    // allow 1 or more
    if (selectedDocIds.length < 1) return;
    setIsCombineMode(true);
    setSelectedDocId("combine");
    setCombineDialogOpen(false);
  }, [selectedDocIds]);

  const handleCombineCancel = useCallback(() => {
    setCombineDialogOpen(false);
    if (!isCombineMode && !selectedDocId && documents[0]) {
      setSelectedDocId(documents[0].documentId);
    }
  }, [documents, isCombineMode, selectedDocId]);

  const clearContext = useCallback(() => {
    setIsCombineMode(false);
    setSelectedDocIds([]);
    setSelectedDocId(documents[0]?.documentId || "");
    clearImages();
  }, [documents, clearImages]);

  const canSend =
    (inputText.trim() !== "" || images.length > 0) &&
    (images.length > 0 || (isCombineMode ? selectedDocIds.length > 0 : !!selectedDocId)) &&
    !isLoading;

  // for optional "Thinking…" bubble if manager didn't push one
  const lastMsg = messages[messages.length - 1];
  const showLoadingThinking = isLoading && !(lastMsg && lastMsg.sender === "ai" && isThinkingMsg(lastMsg));

  // Sum reduced image sizes for quick feedback
  const totalImageKB = useMemo(
    () => Math.round(images.reduce((s, f) => s + f.size, 0) / 1024),
    [images]
  );

  // NEW: Export cards (Excel) for this chat
  const handleExportCards = useCallback(async () => {
    if (!chatId) {
      alert("Chat ID is missing for export.");
      return;
    }
    try {
      setDownloading(true);
      const res = await fetch(`/api/chats/${encodeURIComponent(chatId)}/cards/export`);
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || "Export failed");
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      const base = (chatTitle || chatId || "chat").replace(/[^\w\-]+/g, "_");
      a.href = url;
      a.download = `${base}-cards.xlsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert(err?.message || "Export failed");
    } finally {
      setDownloading(false);
    }
  }, [chatId, chatTitle]);

  // ---- UI ----
  return (
    <div
      className={`flex flex-col h-full bg-white`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
    >
      {/* drag overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-40 pointer-events-none">
          <div className="w-full h-full bg-blue-500/5 border-2 border-dashed border-blue-400" />
        </div>
      )}

      {/* messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => {
          const isAI = message.sender === "ai";

          // prefer array; fall back to single fields; dedup
          const imagesToShow = Array.from(
            new Set(
              [
                ...(Array.isArray(message.imageUrls) ? message.imageUrls.filter(Boolean) : []),
                ...(message.imageUrl ? [message.imageUrl] : []),
                ...(message.plotImageUrl ? [message.plotImageUrl] : []),
              ].filter(Boolean)
            )
          );

          return (
            <div
              key={message.id}
              className={`flex items-start space-x-3 animate-slide-up ${isAI ? "" : "flex-row-reverse space-x-reverse"}`}
            >
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full ${
                  isAI ? "bg-blue-600 shadow-glow" : "bg-gray-200"
                }`}
                aria-hidden
              >
                {isAI ? <Bot className="w-5 h-5 text-white" /> : <User className="w-5 h-5 text-gray-800" />}
              </div>

              <div
                className={`max-w-[70%] p-4 rounded-lg shadow-soft break-words ${
                  isAI ? "bg-blue-50 border border-blue-200" : "bg-gray-100 text-gray-800"
                }`}
              >
                {!!imagesToShow.length && (
                  <div className={`grid gap-2 mb-3 ${isAI ? "grid-cols-1" : imagesToShow.length > 1 ? "grid-cols-2" : "grid-cols-1"}`}>
                    {imagesToShow.map((u, i) => (
                      <img
                        key={`${u}-${i}`}
                        src={u}
                        alt={message.imageAlt || "attachment"}
                        loading="lazy"
                        className={
                          isAI
                            ? "rounded border w-full max-w-[1100px] h-auto object-contain bg-white"
                            : "rounded border max-w-[320px] max-h-64 w-auto h-auto object-contain bg-white"
                        }
                      />
                    ))}
                  </div>
                )}

                {isAI ? (
                  <div className="whitespace-pre-wrap">
                    <ReactMarkdown
                      components={{
                        img: (props: any) => (
                          <img
                            {...props}
                            loading="lazy"
                            className={`w-full max-w-[1100px] h-auto object-contain rounded border my-2 bg-white ${props?.className || ""}`}
                          />
                        ),
                      }}
                    >
                      {message.text}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.text}</p>
                )}

                <p className={`text-xs mt-2 ${isAI ? "text-blue-600" : "text-gray-600"}`}>
                  {new Date(message.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          );
        })}

        {showLoadingThinking && (
          <div className="flex items-start space-x-3" aria-live="polite" aria-busy="true">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-600">
              <Bot className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div className="max-w-[75%] p-4 rounded-lg border bg-white">
              <div className="flex items-center gap-2 text-gray-500">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2"></span>
                </span>
                Thinking…
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* controls */}
      <div className="p-4 border-t border-gray-200 bg-white">
        {/* row 1: select + Export */}
        <div className="flex items-center gap-2 mb-2">
          <select
            className="px-2 py-1 border rounded text-sm bg-white text-gray-800"
            value={isCombineMode ? "combine" : selectedDocId || ""}
            onChange={(e) => handleDropdownChange(e.target.value)}
            disabled={isLoading || documents.length === 0}
          >
            <option value="" disabled>
              Select a file
            </option>
            <option value="combine">🔗 Combine files</option>
            {documents.map((doc) => (
              <option key={doc.documentId} value={doc.documentId}>
                {doc.name}
              </option>
            ))}
          </select>

          {images.length > 0 && (
            <span className="text-[11px] text-gray-600">
              {images.length} file{images.length > 1 ? "s" : ""} • {totalImageKB} KB
            </span>
          )}

          {/* NEW: Export cards (Excel) for this chat */}
          <div className="ml-auto">
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportCards}
              disabled={!chatId || isLoading || downloading}
              title="Download all business-card data uploaded in this chat as Excel"
              aria-label="Export cards to Excel"
              className="border-gray-300 text-gray-700"
            >
              <FileText className="w-3 h-3 mr-2" />
              {downloading ? "Exporting…" : "Export cards"}
            </Button>
          </div>
        </div>

        {/* row 2: chips + attach + previews + Clear */}
        <div className="mb-2 flex items-center gap-2 flex-wrap">
          {chipNames.length > 0 && (
            <>
              <span className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded bg-gray-100 text-gray-700 border">
                <LinkIcon className="w-3 h-3" />
                Using:
              </span>
              <div className="flex items-center gap-2 flex-wrap max-h-20 overflow-y-auto">
                {chipNames.map((n, i) => (
                  <span
                    key={`${n}-${i}`}
                    className="inline-flex items-center text-[11px] px-2 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200"
                  >
                    {n}
                  </span>
                ))}
              </div>
            </>
          )}

          <input
            ref={imgInputRef}
            type="file"
            accept="image/*,.pdf"
            multiple
            className="hidden"
            onChange={(e) => addImages(e.target.files)}
            disabled={isLoading}
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => imgInputRef.current?.click()}
            disabled={isLoading}
            title="Attach image(s) or PDF"
            aria-label="Attach image(s) or PDF"
          >
            {/* <ImagePlus className="w-4 h-4 mr-1" /> */}
            📷
          </Button>

          {/* image previews */}
          {previews.length > 0 && (
            <div className="flex items-center gap-2 overflow-x-auto">
              {previews.map((u, i) => (
                <div key={u} className="relative w-[50px] h-[40px] rounded border overflow-hidden">
                  <img src={u} alt={`preview-${i}`} className="w-full h-full object-cover" />
                  <button
                    type="button"
                    onClick={() => removePreview(i)}
                    className="absolute -top-2 -right-2 bg-white rounded-full shadow p-0.5"
                    title="Remove"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* PDF chips */}
          {pdfNames.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap">
              {pdfNames.map((name, i) => (
                <span
                  key={`${name}-${i}`}
                  className="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-full bg-amber-50 text-amber-700 border border-amber-200"
                  title={name}
                >
                  <FileText className="w-3 h-3" />
                  <span className="max-w-[120px] truncate">{name}</span>
                  <button
                    type="button"
                    onClick={() => removePdf(i)}
                    className="ml-1 rounded-full hover:bg-white/60"
                    aria-label={`Remove ${name}`}
                    title="Remove"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </span>
              ))}
            </div>
          )}

          <button
            type="button"
            onClick={clearContext}
            className="ml-auto inline-flex items-center gap-1 text-sm text-gray-600 hover:text-gray-800"
            title="Clear selection & files"
            disabled={isLoading}
          >
            <X className="w-4 h-4" />
            Clear
          </button>
        </div>

        {/* input row */}
        <form onSubmit={handleSubmit} className="flex gap-3">
          <div className="flex-1 relative">
            <Textarea
              ref={textareaRef}
              value={inputText}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              onCompositionStart={() => (composingRef.current = true)}
              onCompositionEnd={() => (composingRef.current = false)}
              onPaste={handlePaste}
              placeholder="Ask anything about your file(s) — or attach images/PDF…"
              className="min-h-[2.5rem] max-h-[120px] resize-none pr-12 border-gray-300 focus:border-blue-500 focus:ring-blue-200"
              rows={1}
              disabled={isLoading}
              aria-label="Ask a question"
            />
            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <Sparkles className="w-4 h-4 text-gray-400" aria-hidden />
            </div>
          </div>

          <Button
            type="submit"
            disabled={!canSend}
            className="bg-blue-600 hover:bg-blue-700 text-white transition-all duration-200 disabled:opacity-50"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </Button>
        </form>
      </div>

      {/* combine dialog */}
      <Dialog
        open={combineDialogOpen}
        onOpenChange={(open) => (open ? setCombineDialogOpen(true) : handleCombineCancel())}
      >
        <DialogContent className="max-w-md bg-white">
          <DialogHeader>
            <DialogTitle id="combine-dialog-title" className="text-blue-700">
              Select files to combine
            </DialogTitle>
            <DialogDescription id="combine-dialog-description">
              Choose one or more files to use together for your next question.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-2 max-h-[260px] overflow-y-auto">
            {documents.length === 0 && (
              <p className="text-xs text-gray-500">No files available in this chat.</p>
            )}
            {documents.map((doc) => {
              const checked = selectedDocIds.includes(doc.documentId);
              return (
                <label key={doc.documentId} className="flex items-center space-x-2">
                  <Checkbox
                    checked={checked}
                    onCheckedChange={(val) => toggleCombineDoc(doc.documentId, val as boolean)}
                    className="border-gray-300"
                  />
                  <span className="text-sm text-gray-800">{doc.name}</span>
                </label>
              );
            })}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={handleCombineCancel} className="border-gray-300 text-gray-700">
              Cancel
            </Button>
            <Button
              onClick={handleCombineConfirm}
              disabled={selectedDocIds.length < 1}
              className="bg-blue-600 text-white hover:bg-blue-700"
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
