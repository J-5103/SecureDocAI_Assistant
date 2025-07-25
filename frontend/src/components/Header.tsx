import { FileText, Brain, Sparkles } from "lucide-react";

export const Header = () => {
  return (
    <header className="h-16 bg-gradient-primary border-b border-border shadow-soft">
      <div className="h-full px-6 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-10 h-10 bg-accent rounded-lg shadow-glow animate-glow-pulse">
            <Brain className="w-6 h-6 text-accent-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-primary-foreground">SecureDocAI</h1>
            <p className="text-sm text-primary-foreground/80">Intelligent Document Analysis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 px-3 py-2 bg-card/10 rounded-lg">
            <FileText className="w-4 h-4 text-primary-foreground/80" />
            <span className="text-sm text-primary-foreground/80">Secure • Private • AI-Powered</span>
          </div>
          
          <div className="flex items-center space-x-1 text-primary-foreground/60">
            <Sparkles className="w-4 h-4" />
            <span className="text-sm">v1.0</span>
          </div>
        </div>
      </div>
    </header>
  );
};