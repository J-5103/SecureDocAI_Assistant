// src/components/ui/chat/Chat.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
} from '@mui/material';
import ImageIcon from '@mui/icons-material/Image';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';

type ServerAttachment = { filename: string; url: string };

type ServerMsg = {
  message_id: string;
  chat_id: string;
  text: string;
  attachments: ServerAttachment[];
  created_at: string;
};

type UiAttachment = {
  id: string;
  url: string;
  name: string;
};

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
  /** "doc" => /api/ask ;  "viz" => /api/viz/ask  */
  mode: 'doc' | 'viz';
  /** optional: when you want to bind the question to a specific doc/file */
  documentId?: string; // for doc: request.document_id, for viz: file_name or leave empty (backend auto-picks latest)
  height?: number | string;
};

const MAX_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_MIMES = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];

// Inline API base to avoid extra imports
const API_BASE: string =
  (import.meta as any)?.env?.VITE_API_URL ??
  (globalThis as any)?.process?.env?.REACT_APP_API_URL ??
  'http://localhost:8000';

function fmtTime(iso?: string) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function b64ToDataUrl(b64: string, mime = 'image/png') {
  return `data:${mime};base64,${b64}`;
}

export default function Chat({ chatId, mode, documentId, height = '100%' }: Props) {
  const [messages, setMessages] = useState<UiMsg[]>([]);
  const [text, setText] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [sending, setSending] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // Load existing chat history (user-side messages saved via /api/chat)
  const loadHistory = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/chat/history`, { params: { chat_id: chatId } });
      const arr = (res.data?.messages || []) as ServerMsg[];
      arr.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());

      const ui: UiMsg[] = arr.map((m) => ({
        id: m.message_id,
        role: 'user',
        text: m.text,
        attachments: (m.attachments || []).map(a => ({
          id: crypto.randomUUID(),
          name: a.filename,
          url: a.url,
        })),
        created_at: m.created_at,
      }));
      setMessages(ui);
    } catch (e) {
      console.error('History load failed', e);
    }
  }, [chatId]);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  // File picking + previews
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
      urls.push(URL.createObjectURL(f));
    }
    setFiles(prev => [...prev, ...accepted]);
    setPreviews(prev => [...prev, ...urls]);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

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

  // Convert ServerMsg -> UiMsg
  const toUi = (m: ServerMsg): UiMsg => ({
    id: m.message_id,
    role: 'user',
    text: m.text,
    attachments: (m.attachments || []).map(a => ({
      id: crypto.randomUUID(),
      name: a.filename,
      url: a.url,
    })),
    created_at: m.created_at,
  });

  // Model call after upload (creates assistant message)
  const callModel = useCallback(async (questionText: string) => {
    try {
      if (mode === 'doc') {
        const payload = {
          chat_id: chatId,
          question: questionText,
          document_id: documentId || null,
          intent: undefined as any, // free-form, backend will decide
        };
        const res = await axios.post(`${API_BASE}/api/ask`, payload);
        const ans = res.data?.answer ?? '';
        const asst: UiMsg = {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: String(ans || ''),
          attachments: [],
          created_at: new Date().toISOString(),
        };
        setMessages(prev => [...prev, asst]);
      } else {
        // viz
        const payload = {
          chat_id: chatId,
          question: questionText,
          file_name: documentId || undefined, // backend will auto-pick latest if not provided
        };
        const res = await axios.post(`${API_BASE}/api/viz/ask`, payload);
        const ans = res.data?.answer ?? '';
        const imgB64 = res.data?.image_base64 as string | undefined;

        const asst: UiMsg = {
          id: crypto.randomUUID(),
          role: 'assistant',
          text: String(ans || ''),
          attachments: imgB64 ? [{ id: crypto.randomUUID(), name: 'chart.png', url: b64ToDataUrl(imgB64, 'image/png') }] : [],
          created_at: new Date().toISOString(),
        };
        setMessages(prev => [...prev, asst]);
      }
    } catch (e: any) {
      console.error('Model call failed', e);
      const asstErr: UiMsg = {
        id: crypto.randomUUID(),
        role: 'assistant',
        text: e?.response?.data?.error || 'Model call failed.',
        created_at: new Date().toISOString(),
      };
      setMessages(prev => [...prev, asstErr]);
    }
  }, [mode, chatId, documentId]);

  // Send flow: (1) show optimistic, (2) upload to /api/chat, (3) append saved, (4) call model -> append assistant
  const send = useCallback(async () => {
    const q = text.trim();
    if (!q && previews.length === 0) return;

    setSending(true);

    // 1) Optimistic user message (with local preview URLs)
    const optimisticId = crypto.randomUUID();
    const optimistic: UiMsg = {
      id: optimisticId,
      role: 'user',
      text: q,
      attachments: previews.map((url, i) => ({ id: crypto.randomUUID(), name: files[i]?.name || `image-${i+1}`, url })),
      created_at: new Date().toISOString(),
      pending: true,
    };
    setMessages(prev => [...prev, optimistic]);

    try {
      // 2) Upload to /api/chat (saves + returns public URLs)
      const fd = new FormData();
      fd.append('chat_id', chatId);
      fd.append('text', q);
      files.forEach(f => fd.append('files', f, f.name));
      const res = await axios.post(`${API_BASE}/api/chat`, fd);
      const saved = res.data as ServerMsg;

      // Replace optimistic with server version
      setMessages(prev => prev.map(m => (m.id === optimisticId ? toUi(saved) : m)));

      // 3) Clear composer / previews
      previews.forEach(u => URL.revokeObjectURL(u));
      setPreviews([]);
      setFiles([]);
      setText('');

      // 4) Call model for answer
      await callModel(q);
    } catch (e) {
      console.error('Send failed', e);
      // remove optimistic on failure
      setMessages(prev => prev.filter(m => m.id !== optimisticId));
      alert('Failed to send. Check API URL / CORS / server.');
    } finally {
      setSending(false);
    }
  }, [chatId, text, files, previews, callModel]);

  // drag & drop
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
      urls.push(URL.createObjectURL(f));
    }
    if (added.length) {
      setFiles(prev => [...prev, ...added]);
      setPreviews(prev => [...prev, ...urls]);
    }
  }, []);
  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  const canSend = useMemo(() => Boolean(text.trim() || previews.length), [text, previews]);

  return (
    <Stack sx={{ height, p: 2, gap: 1.5 }} onDrop={onDrop} onDragOver={onDragOver}>
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={1} sx={{ pb: 0.5 }}>
        <Avatar sx={{ width: 28, height: 28 }}>{mode === 'viz' ? 'V' : 'D'}</Avatar>
        <Typography variant="subtitle1" fontWeight={600}>
          {mode === 'viz' ? 'Visualization Chat' : 'Document Chat'}
        </Typography>
        <Typography variant="caption" sx={{ ml: 1, opacity: 0.7 }}>#{chatId}</Typography>
      </Stack>

      {/* Messages */}
      <Box sx={{ flex: 1, overflowY: 'auto', pr: 0.5 }}>
        <Stack spacing={1.25}>
          {messages.map(m => {
            const isUser = m.role === 'user';
            return (
              <Stack key={m.id} direction="row" justifyContent={isUser ? 'flex-end' : 'flex-start'}>
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
                        <Box key={a.id} sx={{ width: 120, height: 90, borderRadius: 1, overflow: 'hidden', bgcolor: 'black' }}>
                          <img src={a.url} alt={a.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        </Box>
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
              <img src={url} alt={`preview-${i}`} style={{ width: 120, height: 90, objectFit: 'cover', borderRadius: 8 }} />
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
