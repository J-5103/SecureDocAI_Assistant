import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChatInterface } from "@/components/ChatInterface";
import { PlotsSection } from "@/components/PlotsSection";
import { Header } from "@/components/Header";
import { MessageCircle, Users, BarChart3 } from "lucide-react";

export interface Document {
  id: string;
  name?: string;
  type?: 'pdf' | 'excel' | 'word' | string;
  size?: string;
  status: 'uploaded' | 'processing' | 'ready';
  uploadDate: string;
  chatId?: string; // ✅ supports per-chat filtering if needed later
}

export interface Plot {
  id: string;
  title: string;
  type: string;
  data: any;
  createdAt: string;
}

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: string;
}

const Index = () => {
  const navigate = useNavigate();

  const [plots, setPlots] = useState<Plot[]>([]);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! Welcome to SecureDocAI. Create a chat session to start analyzing your documents, or use the quick chat for temporary conversations.',
      sender: 'ai',
      timestamp: new Date().toISOString()
    }
  ]);

  const [selectedView, setSelectedView] = useState<'chat' | 'plots'>('chat');

  const handleSendMessage = (text: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      text,
      sender: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);

    // Simulate AI response
    setTimeout(() => {
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `This is a quick chat response to: "${text}". For document analysis and persistent conversations, please create a dedicated chat session.`,
        sender: 'ai',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, aiMessage]);

      if (text.toLowerCase().includes('plot') || text.toLowerCase().includes('chart') || text.toLowerCase().includes('graph')) {
        const newPlot: Plot = {
          id: Date.now().toString() + '_plot',
          title: `Quick Analysis - ${text.slice(0, 30)}...`,
          type: 'bar',
          data: {},
          createdAt: new Date().toISOString()
        };
        setPlots(prev => [...prev, newPlot]);
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-secondary">
      <Header />

      <main className="container mx-auto p-6">
        {/* Welcome */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-4">Welcome to SecureDocAI</h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Choose how you'd like to interact with your documents. Create dedicated chat sessions 
            for persistent conversations or use quick chat for temporary queries.
          </p>
        </div>

        {/* Main Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <button
            onClick={() => navigate('/chats')}
            className="group p-8 bg-gradient-primary text-primary-foreground rounded-xl hover:shadow-glow transition-all duration-300 text-left transform hover:scale-105"
          >
            <div className="flex items-center space-x-4 mb-4">
              <div className="p-3 bg-primary-foreground/20 rounded-xl">
                <MessageCircle className="w-8 h-8" />
              </div>
              <h3 className="text-2xl font-bold">Chat Sessions</h3>
            </div>
            <p className="text-lg text-primary-foreground/90 leading-relaxed">
              Create dedicated chats for specific documents. Upload files, analyze content, 
              and continue conversations across sessions with full history.
            </p>
          </button>

          <button
            onClick={() => setSelectedView('chat')}
            className={`group p-8 rounded-xl transition-all duration-300 text-left transform hover:scale-105 border-2 ${
              selectedView === 'chat'
                ? 'bg-accent text-accent-foreground border-accent shadow-glow'
                : 'bg-card text-card-foreground border-border hover:border-accent/50 hover:shadow-card'
            }`}
          >
            <div className="flex items-center space-x-4 mb-4">
              <div className="p-3 bg-accent/20 rounded-xl">
                <Users className="w-8 h-8 text-accent" />
              </div>
              <h3 className="text-2xl font-bold">Quick Chat</h3>
            </div>
            <p className="text-lg opacity-90 leading-relaxed">
              Start a temporary chat session for quick queries. No document upload required — 
              perfect for general AI assistance.
            </p>
          </button>

          <button
            onClick={() => setSelectedView('plots')}
            className={`group p-8 rounded-xl transition-all duration-300 text-left transform hover:scale-105 border-2 ${
              selectedView === 'plots'
                ? 'bg-accent text-accent-foreground border-accent shadow-glow'
                : 'bg-card text-card-foreground border-border hover:border-accent/50 hover:shadow-card'
            }`}
          >
            <div className="flex items-center space-x-4 mb-4">
              <div className="p-3 bg-accent/20 rounded-xl">
                <BarChart3 className="w-8 h-8 text-accent" />
              </div>
              <h3 className="text-2xl font-bold">Visualizations</h3>
            </div>
            <p className="text-lg opacity-90 leading-relaxed">
              View and manage generated plots and charts from your conversations. 
              Currently showing {plots.length} visualization{plots.length !== 1 ? 's' : ''}.
            </p>
          </button>
        </div>

        {/* Dynamic Area */}
        {(selectedView === 'chat' || selectedView === 'plots') && (
          <div className="bg-card rounded-xl border border-border shadow-card overflow-hidden">
            <div className="p-6 border-b border-border">
              <div className="flex space-x-2">
                <button
                  onClick={() => setSelectedView('chat')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                    selectedView === 'chat'
                      ? 'bg-gradient-primary text-primary-foreground shadow-glow'
                      : 'bg-secondary text-secondary-foreground hover:bg-muted'
                  }`}
                >
                  Quick Chat
                </button>
                <button
                  onClick={() => setSelectedView('plots')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                    selectedView === 'plots'
                      ? 'bg-gradient-accent text-accent-foreground shadow-glow'
                      : 'bg-secondary text-secondary-foreground hover:bg-muted'
                  }`}
                >
                  Plots ({plots.length})
                </button>
              </div>
            </div>

            <div className="h-96">
              {selectedView === 'chat' ? (
                <ChatInterface messages={messages} onSendMessage={handleSendMessage} />
              ) : (
                <PlotsSection plots={plots} />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Index;
