// vite.config.ts
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const DEV_HOST = env.VITE_DEV_HOST?.trim() || "0.0.0.0";
  const DEV_PORT = Number(env.VITE_DEV_PORT || 3000);

  // Normalize backend & ollama URLs (no trailing slash)
  const normalize = (u?: string) => (u ? u.replace(/\/+$/, "") : u);

  const BACKEND =
    normalize(env.VITE_BACKEND?.trim()) ||
    normalize(env.VITE_API_BASE?.trim()) ||
    "http://192.168.0.109:8000";

  const OLLAMA = normalize(env.VITE_OLLAMA?.trim() || "http://192.168.0.88:11434");

  console.log("[vite] DEV_HOST:", DEV_HOST, "DEV_PORT:", DEV_PORT);
  console.log("[vite] BACKEND:", BACKEND);
  console.log("[vite] OLLAMA:", OLLAMA);

  const proxyCommon = {
    changeOrigin: true,
    secure: false,
  } as const;

  // Helper to build proxy map (dev & preview both use this)
  const buildProxy = () => ({
    // Main API — forward as-is
    "/api": {
      target: BACKEND,
      ...proxyCommon,
    },
    // Safety net: if UI accidentally calls /api/quickchat (singular),
    // rewrite it to the correct /api/quick-chats before forwarding.
    "/api/quickchat": {
      target: BACKEND,
      ...proxyCommon,
      rewrite: (p: string) => p.replace(/^\/api\/quickchat\b/, "/api/quick-chats"),
    },
    // Static served by backend
    "/static": {
      target: BACKEND,
      ...proxyCommon,
    },
    // Optional Ollama proxy
    "/ollama": {
      target: OLLAMA,
      ...proxyCommon,
      rewrite: (p: string) => p.replace(/^\/ollama/, ""),
    },
  });

  return {
    server: {
      host: DEV_HOST,
      port: DEV_PORT,
      strictPort: true,
      cors: true,
      hmr: {
        host: env.VITE_HMR_HOST?.trim() || undefined,
        clientPort: env.VITE_HMR_CLIENT_PORT
          ? Number(env.VITE_HMR_CLIENT_PORT)
          : undefined,
      },
      proxy: buildProxy(),
    },

    preview: {
      host: "0.0.0.0",
      port: Number(env.VITE_PREVIEW_PORT || 4173),
      proxy: buildProxy(),
    },

    plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),

    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },

    define: {
      // Keep an absolute API base available to the app at build-time
      "import.meta.env.VITE_API_BASE": JSON.stringify(
        env.VITE_API_BASE?.trim() || BACKEND
      ),
    },

    build: {
      sourcemap: mode === "development",
    },
  };
});
