// src/routes/index.tsx
import { Routes, Route, Navigate } from "react-router-dom";
import QuickChat from "@/pages/QuickChat";

export default function AppRoutes() {
  return (
    <Routes>
      {/* Quick Chat list/shell */}
      <Route path="/quick-chat" element={<QuickChat />} />
      {/* Quick Chat detail with chatId in URL */}
      <Route path="/quick-chat/:chatId" element={<QuickChat />} />
      {/* Fallback: redirect everything else */}
      <Route path="*" element={<Navigate to="/quick-chat" replace />} />
    </Routes>
  );
}
