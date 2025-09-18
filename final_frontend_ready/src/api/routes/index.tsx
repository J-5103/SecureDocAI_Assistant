// src/routes/index.js
import { Routes, Route, Navigate } from "react-router-dom";
import QuickChat from "@/pages/QuickChat";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/quick-chat" element={<QuickChat />} />
      {/* Fallback: send everything else to /quick-chat */}
      <Route path="*" element={<Navigate to="/quick-chat" replace />} />
    </Routes>
  );
}
