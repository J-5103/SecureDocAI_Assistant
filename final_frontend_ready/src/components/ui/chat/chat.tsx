// src/components/ui/chat/Chat.tsx
"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useLayoutEffect,
} from 'react';
import axios from 'axios';
import {
  Box,
  Stack,
  TextField,
  IconButton,
  Button,
  Avatar,
  Typography,
  CircularProgress,
  Paper,
  Tooltip,
  Link,
  Chip,
} from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import { useLocation } from 'react-router-dom';

/* --------------------- Types --------------------- */
type ServerAttachment = { filename: string; url: string };
type ServerMsg = {
  message_id: string;
  chat_id: string;
  text: string;
  attachments: ServerAttachment[];
  created_at: string;
};
type UiAttachment = { id: string; url: string; name: string };
type UiMsg = {
  id: string;
  role: 'user' | 'assistant';
  text?: string;
  attachments?: UiAttachment[];
  created_at: string;
  pending?: boolean;
};
type Props = {
  chatId: string;
  mode: 'doc' | 'viz';
  documentId?: string;
  height?: number | string;
  /** Optional: selector/element if a specific container (not window) should be scrolled along with the list */
  scrollRoot?: string | HTMLElement | null;
  /** Optional: keep for compatibility; if "chat" forces stick-to-bottom even if route is custom */
  page?: 'home' | 'chat';
};

/* --------------------- Consts --------------------- */
const MAX_SIZE = 10 * 1024 * 1024;
const ALLOWED_MIMES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];
const API_BASE: string =
  (import.meta as any)?.env?.VITE_API_URL ??
  (globalThis as any)?.process?.env?.REACT_APP_API_URL ??
  'http://localhost:8000';
const SCROLL_BOTTOM = '__BOTTOM__';

/* --------------------- Utils --------------------- */
function fmtTime(iso?: string) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function b64ToDataUrl(b64: string, mime = 'image/png') {
  return `data:${mime};base64,${b64}`;
}
/** Find nearest scrollable ancestor EXCLUDING window. Returns null if none. */
function getNearestScrollableElement(node: HTMLElement | null): HTMLElement | null {
  let el: HTMLElement | null = node;
  while (el) {
    const st = getComputedStyle(el);
    const oy = st.overflowY;
    const isScrollable = (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight;
    if (isScrollable) return el;
    el = el.parentElement;
  }
  return null;
}
function isNearBottomEl(el: HTMLElement, px = 120) {
  return el.scrollHeight - (el.scrollTop + el.clientHeight) <= px;
}
function scrollElToBottom(el: HTMLElement, smooth = true) {
  el.scrollTo({ top: el.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
}

/* ---------------- Attachment thumbnail ---------------- */
function AttachmentThumb({ a, onPaint }: { a: UiAttachment; onPaint?: () => void }) {
  const [failed, setFailed] = useState(false);
  return (
    <Box sx={{ width: 140, height: 110, borderRadius: 1, overflow: 'hidden', bgcolor: 'black' }}>
      {!failed ? (
        <img
          src={a.url}
          alt={a.name}
          crossOrigin="anonymous"
          loading="lazy"
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
          onLoad={onPaint}
          onError={() => setFailed(true)}
        />
      ) : (
        <Stack
          sx={{ width: '100%', height: '100%', alignItems: 'center', justifyContent: 'center', p: 1, bgcolor: 'background.default' }}
          spacing={0.5}
        >
          <InsertDriveFileIcon fontSize="small" />
          <Chip size="small" variant="outlined" label={a.name || 'attachment'} />
          <Link href={a.url} target="_blank" rel="noreferrer" underline="hover">
            Open
          </Link>
        </Stack>
      )}
    </Box>
  );
}

/* --------------------- Component --------------------- */
export default function Chat({
  chatId,
  mode,
  documentId,
  height = '100vh',
  scrollRoot = null,
  page = 'home',
}: Props) {
  const location = useLocation();
  const [messages, setMessages] = useState<UiMsg[]>([]);
  const [text, setText] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [sending, setSending] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [scrollTargetId, setScrollTargetId] = useState<string | null>(null);

  // We manage scroll ourselves
  useLayoutEffect(() => {
    if ('scrollRestoration' in window.history) window.history.scrollRestoration = 'manual';
  }, []);

  /** Should this instance auto-stick to bottom? */
  const shouldStick = useMemo(() => {
    const path = (location?.pathname || '').toLowerCase();
    const byRoute = /\/visualizations\/chat\/|\/chat(\/|$)/.test(path);
    return page === 'chat' || byRoute; // 'chat' prop overrides; otherwise autodetect by route
  }, [location?.pathname, page]);

  /** Resolve a companion scroller element (NOT window) if present */
  const resolveCompanionScroller = useCallback((): HTMLElement | null => {
    if (scrollRoot instanceof HTMLElement) return scrollRoot;
    if (typeof scrollRoot === 'string') {
      const el = document.querySelector(scrollRoot) as HTMLElement | null;
      if (el) return el;
    }
    // Otherwise, try nearest scrollable ancestor (excluding window)
    return getNearestScrollableElement(listRef.current);
  }, [scrollRoot]);

  /** Always scroll the internal list; optionally a companion scroller element (not window) */
  const scrollListBottom = useCallback((smooth = true) => {
    if (listRef.current) scrollElToBottom(listRef.current, smooth);
    endRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'end' });

    const sc = resolveCompanionScroller();
    if (sc) scrollElToBottom(sc, smooth);

    // retry for late paints
    setTimeout(() => {
      if (listRef.current) scrollElToBottom(listRef.current, false);
      const sc2 = resolveCompanionScroller();
      if (sc2) scrollElToBottom(sc2, false);
    }, 0);
    setTimeout(() => {
      if (listRef.current) scrollElToBottom(listRef.current, false);
      const sc3 = resolveCompanionScroller();
      if (sc3) scrollElToBottom(sc3, false);
    }, 150);
  }, [resolveCompanionScroller]);

  /** Scroll to explicit message or bottom (internal list + companion scroller). Never window. */
  const scrollToTarget = useCallback((targetId: string) => {
    if (!shouldStick) return; // home etc. => do nothing
    const doScroll = () => {
      if (targetId === SCROLL_BOTTOM) {
        scrollListBottom(false);
        return;
      }
      const node = itemRefs.current[targetId];
      if (node) {
        node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        const sc = resolveCompanionScroller();
        if (sc && isNearBottomEl(sc)) scrollElToBottom(sc, true);
      } else {
        scrollListBottom(true);
      }
    };
    requestAnimationFrame(() => requestAnimationFrame(doScroll));
  }, [shouldStick, scrollListBottom, resolveCompanionScroller]);

  useEffect(() => {
    if (!scrollTargetId) return;
    scrollToTarget(scrollTargetId);
    const t = setTimeout(() => setScrollTargetId(null), 0);
    return () => clearTimeout(t);
  }, [scrollTargetId, messages.length, scrollToTarget]);

  /* ------------- Data mapping & load ------------- */
  const toUi = (m: ServerMsg): UiMsg => ({
    id: m.message_id,
    role: 'user',
    text: m.text,
    attachments: (m.attachments || []).map(a => ({ id: crypto.randomUUID(), name: a.filename, url: a.url })),
    created_at: m.created_at,
  });

  const loadHistory = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/chat/history`, { params: { chat_id: chatId } });
      const arr = (res.data?.messages || []) as ServerMsg[];
      arr.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
      setMessages(arr.map(toUi));
      // on refresh: only stick for chat/viz chat routes
      if (shouldStick) {
        setScrollTargetId(SCROLL_BOTTOM);
        scrollListBottom(false);
      }
    } catch (e) {
      console.error('History load failed', e);
    }
  }, [chatId, shouldStick, scrollListBottom]);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // keep pinned if already near bottom (list + companion scroller)
  useEffect(() => {
    if (!shouldStick) return;
    const list = listRef.current;
    if (list && isNearBottomEl(list)) scrollElToBottom(list, true);
    const sc = resolveCompanionScroller();
    if (sc && isNearBottomEl(sc)) scrollElToBottom(sc, true);
  }, [messages.length, shouldStick, resolveCompanionScroller]);

  /* ------------- File picking + previews ------------- */
  const [previewsUrlsToRevoke] = useState<string[]>([]);
  const onPick = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const list = e.target.files;
    if (!list) return;
    const accepted: File[] = [];
    const urls: string[] = [];
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (!ALLOWED_MIMES.includes(f.type)) continue;
      if (f.size > MAX_SIZE) continue;
      accepted.push(f);
      const url = URL.createObjectURL(f);
      urls.push(url);
      previewsUrlsToRevoke.push(url);
    }
    setFiles(prev => [...prev, ...accepted]);
    setPreviews(prev => [...prev, ...urls]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [previewsUrlsToRevoke]);

  useEffect(() => {
    return () => { previewsUrlsToRevoke.forEach(u => URL.revokeObjectURL(u)); };
  }, [previewsUrlsToRevoke]);

  const removePreview = useCallback((idx: number) => {
    setPreviews(prev => {
      const url = prev[idx];
      if (url) URL.revokeObjectURL(url);
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
    setFiles(prev => {
      const copy = [...prev];
      copy.splice(idx, 1);
      return copy;
    });
  }, []);

  /* ------------- Model call (assistant) ------------- */
  const callModel = useCallback(async (questionText: string) => {
    try {
      if (mode === 'doc') {
        const res = await axios.post(`${API_BASE}/api/ask`, {
          chat_id: chatId,
          question: questionText,
          document_id: documentId || null,
          intent: undefined as any,
        });
        const ans = res.data?.answer ?? '';
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: String(ans || ''), attachments: [], created_at: new Date().toISOString() }]);
        if (shouldStick) setScrollTargetId(SCROLL_BOTTOM);
      } else {
        const res = await axios.post(`${API_BASE}/api/viz/ask`, {
          chat_id: chatId,
          question: questionText,
          file_name: documentId || undefined,
        });
        const ans = res.data?.answer ?? '';
        const imgB64 = res.data?.image_base64 as string | undefined;
        setMessages(prev => [...prev, {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: String(ans || ''),
          attachments: imgB64 ? [{ id: crypto.randomUUID(), name: 'chart.png', url: b64ToDataUrl(imgB64, 'image/png') }] : [],
          created_at: new Date().toISOString(),
        }]);
        if (shouldStick) setScrollTargetId(SCROLL_BOTTOM);
      }
    } catch (e: any) {
      console.error('Model call failed', e);
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: e?.response?.data?.error || 'Model call failed.', created_at: new Date().toISOString() }]);
      if (shouldStick) setScrollTargetId(SCROLL_BOTTOM);
    }
  }, [mode, chatId, documentId, shouldStick]);

  /* ------------- Send ------------- */
  const send = useCallback(async () => {
    const q = text.trim();
    if (!q && previews.length === 0) return;

    setSending(true);

    const optimisticId = crypto.randomUUID();
    const optimistic: UiMsg = {
      id: optimisticId, role: 'user', text: q,
      attachments: previews.map((url, i) => ({ id: crypto.randomUUID(), name: files[i]?.name || `image-${i + 1}`, url })),
      created_at: new Date().toISOString(), pending: true,
    };
    setMessages(prev => [...prev, optimistic]);
    if (shouldStick) setScrollTargetId(optimisticId); // focus exact new question

    try {
      const fd = new FormData();
      fd.append('chat_id', chatId);
      fd.append('text', q);
      files.forEach(f => fd.append('files', f, f.name));
      const res = await axios.post(`${API_BASE}/api/chat`, fd);
      const saved = res.data as ServerMsg;

      setMessages(prev => prev.map(m => (m.id === optimisticId ? toUi(saved) : m)));

      previews.forEach(u => URL.revokeObjectURL(u));
      setPreviews([]);
      setFiles([]);
      setText('');

      await callModel(q);
    } catch (e) {
      console.error('Send failed', e);
      setMessages(prev => prev.filter(m => m.id !== optimisticId));
      alert('Failed to send. Check API URL / CORS / server.');
    } finally {
      setSending(false);
    }
  }, [chatId, text, files, previews, callModel, shouldStick]);

  /* ------------- DnD ------------- */
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const dt = e.dataTransfer;
    if (!dt?.files?.length) return;
    const added: File[] = [];
    const urls: string[] = [];
    for (const f of Array.from(dt.files)) {
      if (!ALLOWED_MIMES.includes(f.type)) continue;
      if (f.size > MAX_SIZE) continue;
      added.push(f);
      const url = URL.createObjectURL(f);
      urls.push(url);
      previewsUrlsToRevoke.push(url);
    }
    if (added.length) {
      setFiles(prev => [...prev, ...added]);
      setPreviews(prev => [...prev, ...urls]);
    }
  }, [previewsUrlsToRevoke]);
  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  const canSend = useMemo(() => Boolean(text.trim() || previews.length), [text, previews]);

  /* --------------------- Render --------------------- */
  return (
    <Stack
      sx={{
        height,
        p: 2,
        gap: 1.5,
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
      }}
      onDrop={onDrop}
      onDragOver={onDragOver}
    >
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={1} sx={{ pb: 0.5 }}>
        <Avatar sx={{ width: 28, height: 28 }}>{mode === 'viz' ? 'V' : 'D'}</Avatar>
        <Typography variant="subtitle1" fontWeight={600}>
          {mode === 'viz' ? 'Visualization Chat' : 'Document Chat'}
        </Typography>
        <Typography variant="caption" sx={{ ml: 1, opacity: 0.7 }}>#{chatId}</Typography>
      </Stack>

      {/* Messages */}
      <Box
        ref={listRef}
        sx={{
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
          pr: 0.5,
          overscrollBehavior: 'contain',
          scrollBehavior: 'smooth',
        }}
      >
        <Stack spacing={1.25}>
          {messages.map(m => {
            const isUser = m.role === 'user';
            return (
              <Stack
                key={m.id}
                ref={(el) => { itemRefs.current[m.id] = el; }}
                direction="row"
                justifyContent={isUser ? 'flex-end' : 'flex-start'}
              >
                <Stack
                  component={Paper}
                  elevation={1}
                  sx={{
                    maxWidth: '78%',
                    p: 1.25,
                    bgcolor: isUser ? 'primary.main' : 'background.paper',
                    color: isUser ? 'primary.contrastText' : 'text.primary',
                    borderTopLeftRadius: 12,
                    borderTopRightRadius: 12,
                    borderBottomLeftRadius: isUser ? 12 : 2,
                    borderBottomRightRadius: isUser ? 2 : 12,
                  }}
                >
                  {!!m.attachments?.length && (
                    <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', mb: m.text ? 1 : 0 }}>
                      {m.attachments.map(a => (
                        <AttachmentThumb
                          key={a.id}
                          a={a}
                          onPaint={() => {
                            if (!shouldStick) return;
                            // if list/companion is already near bottom, keep it pinned
                            const list = listRef.current;
                            if (list && isNearBottomEl(list)) requestAnimationFrame(() => scrollElToBottom(list, true));
                            const sc = resolveCompanionScroller();
                            if (sc && isNearBottomEl(sc)) requestAnimationFrame(() => scrollElToBottom(sc, true));
                          }}
                        />
                      ))}
                    </Stack>
                  )}

                  {!!m.text && <Typography variant="body2" whiteSpace="pre-wrap">{m.text}</Typography>}
                  <Stack direction="row" alignItems="center" spacing={1} sx={{ mt: 0.5, opacity: 0.85 }}>
                    {m.pending && <CircularProgress size={14} />}
                    <Typography variant="caption">{fmtTime(m.created_at)}</Typography>
                  </Stack>
                </Stack>
              </Stack>
            );
          })}
          <div ref={endRef} />
        </Stack>
      </Box>

      {/* Previews before send */}
      {!!previews.length && (
        <Stack direction="row" spacing={1} sx={{ overflowX: 'auto' }}>
          {previews.map((url, i) => (
            <Box key={url} sx={{ position: 'relative' }}>
              <img
                src={url}
                alt={`preview-${i}`}
                style={{ width: 120, height: 90, objectFit: 'cover', borderRadius: 8, display: 'block' }}
                onLoad={() => {
                  if (!shouldStick) return;
                  const list = listRef.current;
                  if (list && isNearBottomEl(list)) requestAnimationFrame(() => scrollElToBottom(list, true));
                  const sc = resolveCompanionScroller();
                  if (sc && isNearBottomEl(sc)) requestAnimationFrame(() => scrollElToBottom(sc, true));
                }}
              />
              <IconButton
                size="small"
                onClick={() => removePreview(i)}
                sx={{ position: 'absolute', top: -10, right: -10, bgcolor: 'background.paper' }}
              >
                <CloseIcon fontSize="small" />
              </IconButton>
            </Box>
          ))}
        </Stack>
      )}

      {/* Composer */}
      <Stack direction="row" spacing={1} alignItems="flex-end">
        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_MIMES.join(',')}
          multiple
          hidden
          onChange={onPick}
        />
        <Tooltip title="Attach images">
          <IconButton onClick={() => fileInputRef.current?.click()}>
            <ImageIcon />
          </IconButton>
        </Tooltip>

        <TextField
          fullWidth
          multiline
          minRows={1}
          maxRows={6}
          placeholder="Type your message…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
        />

        <Button variant="contained" endIcon={<SendIcon />} onClick={send} disabled={sending || !canSend}>
          Send
        </Button>
      </Stack>
    </Stack>
  );
}
