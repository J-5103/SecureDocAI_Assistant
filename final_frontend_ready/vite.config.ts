// vite.config.ts
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  // Dev host/port for Vite
  const DEV_HOST = env.VITE_DEV_HOST || "192.168.0.110";
  const DEV_PORT = Number(env.VITE_DEV_PORT || 3000);

  // FastAPI backend base (used by proxy)
  // Accepts either VITE_BACKEND or VITE_API_BASE; falls back to your LAN IP.
  const BACKEND =
    (env.VITE_BACKEND && env.VITE_BACKEND.trim()) ||
    (env.VITE_API_BASE && env.VITE_API_BASE.trim()) ||
    "http://192.168.0.110:8000";

  // Ollama server base (used by proxy for /ollama/*)
  const OLLAMA =
    (env.VITE_OLLAMA && env.VITE_OLLAMA.trim()) ||
    "http://192.168.0.88:11434";

  return {
    server: {
      host: DEV_HOST, // access from LAN
      port: DEV_PORT,
      cors: true,
      hmr: { host: DEV_HOST },
      proxy: {
        // FastAPI (JSON, uploads, etc.)
        "/api": {
          target: BACKEND,
          changeOrigin: true,
          secure: false,
        },
        // Static files served by FastAPI (/static/uploads, viz thumbs, etc.)
        "/static": {
          target: BACKEND,
          changeOrigin: true,
          secure: false,
        },
        // Ollama (model server) — call via /ollama/* from the frontend
        // e.g. fetch('/ollama/api/generate', {...})
        "/ollama": {
          target: OLLAMA,
          changeOrigin: true,
          secure: false,
          rewrite: (p) => p.replace(/^\/ollama/, ""),
        },
      },
    },
    plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    // Provide a sane default for VITE_API_BASE at build time if not set
    define: {
      "import.meta.env.VITE_API_BASE": JSON.stringify(
        env.VITE_API_BASE || BACKEND
      ),
    },
  };
});
