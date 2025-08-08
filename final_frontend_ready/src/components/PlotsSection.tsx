import { BarChart, LineChart, PieChart, TrendingUp, Download, Share2, Trash2 } from "lucide-react";
import { Plot } from "@/pages/Index";
import { Button } from "@/components/ui/button";

interface PlotsSectionProps {
  plots: Plot[];
}

export const PlotsSection = ({ plots }: PlotsSectionProps) => {
  const getPlotIcon = (type: string) => {
    switch (type) {
      case 'bar':
        return <BarChart className="w-5 h-5 text-accent" />;
      case 'line':
        return <LineChart className="w-5 h-5 text-success" />;
      case 'pie':
        return <PieChart className="w-5 h-5 text-primary" />;
      default:
        return <TrendingUp className="w-5 h-5 text-muted-foreground" />;
    }
  };

  if (plots.length === 0) {
    return (
      <div className="h-full flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="w-24 h-24 bg-gradient-accent rounded-full flex items-center justify-center mx-auto mb-6 shadow-glow animate-glow-pulse">
            <TrendingUp className="w-12 h-12 text-accent-foreground" />
          </div>
          <h3 className="text-xl font-semibold text-foreground mb-2">No Plots Generated Yet</h3>
          <p className="text-muted-foreground mb-6 max-w-md">
            Ask the AI to create charts, graphs, or visualizations from your document data.
            All generated plots will appear here for easy access.
          </p>
          <div className="flex flex-wrap justify-center gap-2">
            {[
              "Create a bar chart",
              "Show me a pie chart",
              "Generate a line graph",
              "Plot the trends"
            ].map((suggestion) => (
              <span
                key={suggestion}
                className="px-3 py-1 text-sm bg-secondary rounded-full text-muted-foreground"
              >
                {suggestion}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-background">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Generated Plots</h2>
            <p className="text-muted-foreground">All your AI-generated visualizations</p>
          </div>
          <div className="flex space-x-2">
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export All
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {plots.map((plot) => (
            <div
              key={plot.id}
              className="group bg-card rounded-lg border border-border shadow-card hover:shadow-glow transition-all duration-200 animate-slide-up"
            >
              {/* Plot Preview */}
              <div className="h-48 bg-gradient-secondary rounded-t-lg p-4 flex items-center justify-center">
                <div className="text-center">
                  {getPlotIcon(plot.type)}
                  <p className="text-sm text-muted-foreground mt-2">
                    {plot.type.toUpperCase()} Chart Preview
                  </p>
                  <div className="mt-4 w-32 h-20 bg-accent/10 rounded border-2 border-dashed border-accent/30 flex items-center justify-center">
                    <TrendingUp className="w-8 h-8 text-accent/50" />
                  </div>
                </div>
              </div>

              {/* Plot Info */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-semibold text-foreground truncate mb-1">
                      {plot.title}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {new Date(plot.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex justify-between items-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex space-x-2">
                    <Button variant="outline" size="sm">
                      <Download className="w-3 h-3" />
                    </Button>
                    <Button variant="outline" size="sm">
                      <Share2 className="w-3 h-3" />
                    </Button>
                  </div>
                  <Button variant="destructive" size="sm">
                    <Trash2 className="w-3 h-3" />
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};