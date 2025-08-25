// src/App.tsx
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Index from "./pages/Index";
import { ChatManager } from "./pages/ChatManager";
import NotFound from "./pages/NotFound";

// Visualization chat detail (Excel/CSV)
import { VizChatManager } from "./pages/VizChatManager";

// Visualization chats list (Chat Sessions–style UI)
import Vizchats from "./pages/visualization/Vizchats";

// Optional: visualizations gallery page
import VisualizationPage from "./pages/visualizationpage";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* Home */}
          <Route path="/" element={<Index />} />

          {/* Standard chats (PDF/Docs): list + detail */}
          <Route path="/chats" element={<ChatManager />} />
          <Route path="/chat/:chatId" element={<ChatManager />} />

          {/* Direct alias for doc chat detail */}
          <Route path="/chat/doc/:chatId" element={<ChatManager />} />

          {/* Visualization (Excel/CSV): list + detail */}
          <Route path="/visualizations" element={<Vizchats />} />
          <Route path="/visualizations/chat/:chatId" element={<VizChatManager />} />
          <Route path="/visualizations/gallery" element={<VisualizationPage />} />

          {/* Direct alias for viz chat detail */}
          <Route path="/chat/viz/:chatId" element={<VizChatManager />} />

          {/* --- Backward-compatibility / aliases --- */}
          {/* Old hyphen style */}
          <Route path="/viz-chats" element={<Navigate to="/visualizations" replace />} />
          <Route
            path="/viz-chat/:chatId"
            element={<Navigate to="/visualizations/chat/:chatId" replace />}
          />
          {/* Old slash style */}
          <Route path="/viz/chats" element={<Navigate to="/visualizations" replace />} />
          <Route
            path="/viz/chat/:chatId"
            element={<Navigate to="/visualizations/chat/:chatId" replace />}
          />

          {/* 404 */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
