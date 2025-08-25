export const ChatMessage = ({ message }: { message: { text: string; sender: "user" | "ai"; timestamp: string } }) => {
  const messageClass =
    message.sender === "ai" ? "bg-blue-100 text-blue-800" : "bg-gray-100 text-gray-800"; // Different colors for user and AI

  return (
    <div className={`p-4 rounded-lg ${messageClass}`}>
      <div className="flex items-start space-x-2">
        <span className={`font-semibold ${message.sender === "ai" ? "text-blue-600" : "text-gray-600"}`}>
          {message.sender === "ai" ? "AI" : "You"}:
        </span>
        <p className="text-sm">{message.text}</p>
      </div>
      <div className="text-xs text-gray-500 mt-1">{new Date(message.timestamp).toLocaleTimeString()}</div>
    </div>
  );
};
