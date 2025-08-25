const BASE_URL = process.env.REACT_APP_API_BASE || "http://192.168.0.109:8000";

export async function fetchVisualizations(chat = "") {
  const url = chat
    ? `${BASE_URL}/api/visualizations?chat=${encodeURIComponent(chat)}`
    : `${BASE_URL}/api/visualizations`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to load visualizations");
  const data = await res.json();
  return data.items || [];
}
