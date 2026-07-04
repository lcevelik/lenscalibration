const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getBackendPort: () => ipcRenderer.invoke('get-backend-port'),
  getLocalIP:     () => ipcRenderer.invoke('get-local-ip'),
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
  showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
  // File.path was removed in Electron 32; webUtils.getPathForFile is the
  // supported way to resolve a dropped File to a filesystem path.
  getPathForFile: (file) => webUtils.getPathForFile(file),
});
