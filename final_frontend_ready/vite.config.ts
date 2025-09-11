// vite.config.ts
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  // ---- Dev host/port for Vite (LAN-friendly) ----
  const DEV_HOST = env.VITE_DEV_HOST?.trim() || "0.0.0.0"; // 0.0.0.0 lets other devices connect
  const DEV_PORT = Number(env.VITE_DEV_PORT || 3000);

  // ---- FastAPI backend base (used by proxy & as default API base) ----
  // Accepts VITE_BACKEND or VITE_API_BASE; fallback to your LAN IP:8000.
  const BACKEND =
    env.VITE_BACKEND?.trim() ||
    env.VITE_API_BASE?.trim() ||
    "http://192.168.0.109:8000";

  // ---- Optional: Ollama server (if you call it via /ollama/*) ----
  const OLLAMA = env.VITE_OLLAMA?.trim() || "http://192.168.0.88:11434";

  // Useful note in console at dev start
  console.log("[vite] DEV_HOST:", DEV_HOST, "DEV_PORT:", DEV_PORT);
  console.log("[vite] BACKEND:", BACKEND);
  console.log("[vite] OLLAMA:", OLLAMA);

  const proxyCommon = {
    changeOrigin: true,
    secure: false,
  } as const;

  return {
    server: {
      host: DEV_HOST,
      port: DEV_PORT,
      strictPort: true,
      cors: true,
      hmr: {
        host: env.VITE_HMR_HOST?.trim() || undefined, // set to your IP if needed
        clientPort: env.VITE_HMR_CLIENT_PORT
          ? Number(env.VITE_HMR_CLIENT_PORT)
          : undefined,
      },
      proxy: {
        // FastAPI (JSON, uploads, downloads, export, etc.)
        "/api": {
          target: BACKEND,
          ...proxyCommon,
        },
        // Static files served by FastAPI (/static/uploads, viz images/thumbs, etc.)
        "/static": {
          target: BACKEND,
          ...proxyCommon,
        },
        // Optional Ollama proxy (strip leading /ollama)
        "/ollama": {
          target: OLLAMA,
          ...proxyCommon,
          rewrite: (p) => p.replace(/^\/ollama/, ""),
        },
      },
    },

    // Preview server mirrors dev proxy so exports still work in `vite preview`
    preview: {
      host: "0.0.0.0",
      port: Number(env.VITE_PREVIEW_PORT || 4173),
      proxy: {
        "/api": { target: BACKEND, ...proxyCommon },
        "/static": { target: BACKEND, ...proxyCommon },
        "/ollama": {
          target: OLLAMA,
          ...proxyCommon,
          rewrite: (p) => p.replace(/^\/ollama/, ""),
        },
      },
    },

    plugins: [react(), mode === "development" && componentTagger()].filter(
      Boolean
    ),

    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },

    // Provide a default for VITE_API_BASE at build time if not set
    // This keeps absolute API calls (e.g., buildApiUrl) pointed at FastAPI.
    define: {
      "import.meta.env.VITE_API_BASE": JSON.stringify(
        env.VITE_API_BASE?.trim() || BACKEND
      ),
    },

    // Optional: nicer source maps while debugging
    build: {
      sourcemap: mode === "development",
    },
  };
});
