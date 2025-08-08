export const Header = () => {
  return (
    <header className="h-16 bg-gradient-primary border-b border-border shadow-soft">
      <div className="h-full px-6 flex items-center justify-between">
        {/* Left Side: Logo + App Name */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-10 h-10 bg-white rounded-lg shadow-glow animate-glow-pulse overflow-hidden">
            <img
              src="/images/logo.png"
              alt="App Logo"
              className="w-8 h-8 object-contain"
            />
          </div>
          <div>
            <h1 className="text-xl font-bold text-primary-foreground">Prashi Solution</h1>
            <p className="text-sm text-primary-foreground/80">Intelligent Document Analysis</p>
          </div>
        </div>

        {/* Right side removed */}
      </div>
    </header>
  );
};
