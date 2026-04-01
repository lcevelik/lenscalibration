export {};

declare global {
  interface Window {
    electronAPI?: {
      getBackendPort: () => Promise<number>;
      showSaveDialog: (options: {
        defaultPath?: string;
        filters?: Array<{ name: string; extensions: string[] }>;
      }) => Promise<{ canceled: boolean; filePath?: string }>;
    };
  }
}
