import { useState } from "react";
import { Upload, FileText, File, FileSpreadsheet, Trash2, Eye } from "lucide-react";
import { Document } from "@/pages/Index";
import { useToast } from "@/hooks/use-toast";
import { uploadFile } from "@/api/document_API"; // ✅ Import your upload API



interface DocumentSidebarProps {
  documents: Document[];
  onDocumentUpload: (files: FileList) => void;
}

export const DocumentSidebar = ({ documents, onDocumentUpload }: DocumentSidebarProps) => {
  const [dragActive, setDragActive] = useState(false);
  const { toast } = useToast();

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  const handleFileInput = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const validFiles = e.target.files;

      // Optional: Send raw files to local state via onDocumentUpload (frontend only)
      onDocumentUpload(validFiles);

      try {
        // Upload each file to backend
        for (let i = 0; i < validFiles.length; i++) {
          const file = validFiles[i];

          await uploadFile(file); // Upload to backend (no need to create extra object here)
        }

        toast({
          title: "Upload Successful",
          description: `${validFiles.length} file(s) uploaded to backend.`,
        });

      } catch (error: any) {
        console.error("❌ Upload Error:", error.message);
        toast({
          title: "Upload Failed",
          description: error.message || "An error occurred during upload.",
        });
      }
    }
  };




  // ✅ Handle upload to backend + update frontend state
  const handleFileUpload = async (fileList: FileList) => {
    const validFiles = filterValidFiles(fileList);

    if (validFiles.length === 0) {
      toast({
        title: "Upload Error",
        description: "Only PDF, Word, and Excel files are supported.",
      });
      return;
    }

    try {
      for (const file of Array.from(validFiles)) {
        const response = await uploadFile(file);
        console.log("📄 Uploaded to:", response.document_path);
      }

      // After uploading to backend, update frontend state
      onDocumentUpload(validFiles);

      toast({
        title: "Upload Successful",
        description: `${validFiles.length} file(s) uploaded.`,
      });

    } catch (err: any) {
      console.error(err);
      toast({
        title: "Upload Error",
        description: "Failed to upload document to server.",
      });
    }
  };

  const filterValidFiles = (fileList: FileList): FileList => {
    const supportedTypes = [".pdf", ".doc", ".docx", ".xls", ".xlsx"];
    const validFilesArray = Array.from(fileList).filter(file =>
      supportedTypes.some(ext => file.name.toLowerCase().endsWith(ext))
    );

    const dataTransfer = new DataTransfer();
    validFilesArray.forEach(file => dataTransfer.items.add(file));
    return dataTransfer.files;
  };

  const getFileIcon = (type: Document['type']) => {
    switch (type) {
      case 'pdf': return <FileText className="w-5 h-5 text-destructive" />;
      case 'excel': return <FileSpreadsheet className="w-5 h-5 text-success" />;
      case 'word': return <File className="w-5 h-5 text-primary" />;
    }
  };

  const getStatusColor = (status: Document['status']) => {
    switch (status) {
      case 'uploaded': return 'bg-muted text-muted-foreground';
      case 'processing': return 'bg-accent/20 text-accent animate-pulse';
      case 'ready': return 'bg-success/20 text-success';
    }
  };

  return (
    <div className="w-80 bg-card border-r border-border flex flex-col shadow-card">
      {/* Upload Area */}
      <div className="p-4 border-b border-border">
        <div
          className={`border-2 border-dashed rounded-lg p-6 text-center transition-all duration-200 ${
            dragActive ? 'border-accent bg-accent/5 shadow-glow' : 'border-border hover:border-accent/50 hover:bg-accent/5'
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <Upload className={`w-8 h-8 mx-auto mb-3 ${dragActive ? 'text-accent' : 'text-muted-foreground'}`} />
          <p className="text-sm font-medium mb-2">
            Drop files here or{' '}
            <label className="text-accent hover:text-accent-glow cursor-pointer">
              browse
              <input
                type="file"
                multiple
                accept=".pdf,.doc,.docx,.xls,.xlsx"
                className="hidden"
                onChange={handleFileInput}
              />
            </label>
          </p>
          <p className="text-xs text-muted-foreground">PDF, Word, Excel files supported</p>
        </div>
      </div>

      {/* Documents List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <h3 className="text-sm font-semibold text-foreground mb-3">
            Documents ({documents.length})
          </h3>

          {documents.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="w-12 h-12 text-muted-foreground mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">No documents uploaded yet</p>
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="group p-3 bg-secondary rounded-lg hover:bg-muted transition-all duration-200 animate-slide-up"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3 flex-1 min-w-0">
                      {getFileIcon(doc.type)}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-foreground truncate">{doc.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {doc.size} • {new Date(doc.uploadDate).toLocaleDateString()}
                        </p>
                        <span className={`inline-block px-2 py-1 rounded text-xs font-medium mt-1 ${getStatusColor(doc.status)}`}>
                          {doc.status}
                        </span>
                      </div>
                    </div>

                    <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button className="p-1 hover:bg-accent/20 rounded text-accent">
                        <Eye className="w-3 h-3" />
                      </button>
                      <button className="p-1 hover:bg-destructive/20 rounded text-destructive">
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
