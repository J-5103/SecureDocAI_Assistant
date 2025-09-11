// src/components/ExportButton.tsx
import React, { useCallback, useMemo, useState } from "react";
import { openCardsExport, cardsExport, cardsExportUrl } from "@/api/api";

type ExportFormat = "xlsx" | "csv" | "vcf" | "zip";
type ExportMethod = "open" | "download";

export interface ExportButtonProps
  // Avoid conflict with DOM's onError type on buttons
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onError"> {
  chatId: string;
  /** Export file format; default "xlsx" */
  format?: ExportFormat;
  /** "open" = open new tab; "download" = stream blob; default "open" */
  method?: ExportMethod;
  /** Called right before export starts */
  onStart?: () => void;
  /** Called on success */
  onComplete?: (fileName?: string) => void;
  /** Called on failure (renamed to prevent clash with DOM onError) */
  onExportError?: (err: Error) => void;
  /** Custom label; default is based on format */
  label?: React.ReactNode;
}

/** tiny inline SVG to avoid icon deps */
const DownloadIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="currentColor"
    aria-hidden="true"
  >
    <path d="M12 3a1 1 0 0 1 1 1v8.586l2.293-2.293a1 1 0 1 1 1.414 1.414l-4 4a1 1 0 0 1-1.414 0l-4-4A1 1 0 1 1 8.707 10.293L11 12.586V4a1 1 0 0 1 1-1z" />
    <path d="M5 20a2 2 0 0 1-2-2v-2a1 1 0 1 1 2 0v2h14v-2a1 1 0 1 1 2 0v2a2 2 0 0 1-2 2H5z" />
  </svg>
);

export const ExportButton: React.FC<ExportButtonProps> = ({
  chatId,
  format = "xlsx",
  method = "open",
  onStart,
  onComplete,
  onExportError,
  label,
  disabled,
  className,
  style,
  ...btnProps
}) => {
  const [loading, setLoading] = useState(false);

  const computedLabel = useMemo(
    () => label ?? `Export (.${format})`,
    [label, format]
  );

  const handleError = useCallback(
    (err: unknown) => {
      const e = err instanceof Error ? err : new Error(String(err ?? "Error"));
      if (onExportError) onExportError(e);
      else alert(e.message);
    },
    [onExportError]
  );

  const handleClick = useCallback(async () => {
    if (!chatId) {
      handleError(new Error("Chat ID is required for export."));
      return;
    }

    try {
      setLoading(true);
      onStart?.();

      if (method === "open") {
        // Simple new-tab download (no dependencies)
        openCardsExport({ chatId, format });
        onComplete?.();
        return;
      }

      // method === "download" → stream blob and save (no file-saver needed)
      const { blob, filename } = await cardsExport({ chatId, format });

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename || `export.${format}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      onComplete?.(filename);
    } catch (err) {
      handleError(err);
    } finally {
      setLoading(false);
    }
  }, [chatId, format, method, onStart, onComplete, handleError]);

  const isDisabled = disabled || !chatId || loading;

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isDisabled}
      className={className}
      title={
        chatId
          ? `Download ${format.toUpperCase()} for this chat`
          : "Provide a chatId to export"
      }
      {...btnProps}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "8px 12px",
        borderRadius: 8,
        border: "1px solid rgba(0,0,0,0.12)",
        background: isDisabled ? "#e0e0e0" : "#f5f5f5",
        color: "#111",
        cursor: isDisabled ? "not-allowed" : "pointer",
        ...(style || {}),
      }}
    >
      {loading ? (
        <span style={{ fontSize: 12 }}>Exporting…</span>
      ) : (
        <>
          <DownloadIcon />
          <span>{computedLabel}</span>
        </>
      )}
    </button>
  );
};

export default ExportButton;

/* ---------- Optional helper: plain anchor variant ----------
Usage:
  <ExportAnchor chatId={chatId} format="xlsx">Export</ExportAnchor>
This renders an <a> pointing directly at the absolute backend URL.
*/
export const ExportAnchor: React.FC<
  React.AnchorHTMLAttributes<HTMLAnchorElement> & {
    chatId: string;
    format?: ExportFormat;
  }
> = ({ chatId, format = "xlsx", children, ...aProps }) => {
  const href = useMemo(
    () => (chatId ? cardsExportUrl({ chatId, format }) : "#"),
    [chatId, format]
  );
  const disabled = !chatId;

  return (
    <a
      href={disabled ? undefined : href}
      target="_blank"
      rel="noopener"
      aria-disabled={disabled}
      {...aProps}
    >
      {children ?? `Export (.${format})`}
    </a>
  );
};
