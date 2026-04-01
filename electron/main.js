const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const net = require('net');
const path = require('path');

let backendPort = null;
let backendProcess = null;

function getAvailablePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    server.on('error', reject);
  });
}

function startBackend(port) {
  return new Promise((resolve, reject) => {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const backendPath = app.isPackaged
      ? path.join(process.resourcesPath, 'backend', 'main.py')
      : path.join(__dirname, '..', 'backend', 'main.py');

    backendProcess = spawn(pythonCmd, [backendPath, '--port', String(port)], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let resolved = false;

    const startupTimer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        // 4 s passed without the ready message — still try to proceed but warn
        console.warn('Backend startup message not received within 4 s; continuing anyway');
        resolve();
      }
    }, 4000);

    const onReady = (data) => {
      if (!resolved && data.toString().includes('Application startup complete')) {
        resolved = true;
        clearTimeout(startupTimer);
        resolve();
      }
    };

    backendProcess.stdout.on('data', onReady);
    backendProcess.stderr.on('data', onReady);

    backendProcess.on('error', (err) => {
      clearTimeout(startupTimer);
      if (!resolved) { resolved = true; reject(err); }
    });

    // Detect backend crash after it has started
    backendProcess.on('exit', (code, signal) => {
      clearTimeout(startupTimer);
      if (!resolved) {
        resolved = true;
        reject(new Error(`Backend exited during startup (code ${code}, signal ${signal})`));
      } else {
        console.error(`Backend process exited unexpectedly (code ${code}, signal ${signal})`);
      }
    });
  });
}

/** Gracefully shut down the backend: SIGTERM, then SIGKILL after 3 s. */
function stopBackend() {
  return new Promise((resolve) => {
    if (!backendProcess) { resolve(); return; }
    const proc = backendProcess;
    backendProcess = null;

    const forceKill = setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch (_) {}
      resolve();
    }, 3000);

    proc.on('exit', () => {
      clearTimeout(forceKill);
      resolve();
    });

    try { proc.kill('SIGTERM'); } catch (_) { clearTimeout(forceKill); resolve(); }
  });
}

async function createWindow() {
  backendPort = await getAvailablePort();
  await startBackend(backendPort);

  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (app.isPackaged) {
    win.loadFile(path.join(__dirname, '..', 'frontend', 'dist', 'index.html'));
  } else {
    win.loadURL('http://localhost:5173');
    win.webContents.openDevTools();
  }
}

ipcMain.handle('get-backend-port', () => backendPort);

ipcMain.handle('show-save-dialog', async (_, options) => {
  // Whitelist allowed option keys to prevent unexpected Electron API behaviour
  const safeOptions = {};
  if (options && typeof options.defaultPath === 'string') safeOptions.defaultPath = options.defaultPath;
  if (options && Array.isArray(options.filters)) safeOptions.filters = options.filters;
  try {
    return await dialog.showSaveDialog(safeOptions);
  } catch (err) {
    console.error('showSaveDialog error:', err);
    return { canceled: true, filePath: undefined };
  }
});

app.whenReady().then(createWindow).catch((err) => {
  console.error('Failed to start application:', err);
  dialog.showErrorBox('Startup Error', `Failed to start Lens Calibration:\n\n${err.message}`);
  app.quit();
});

let isQuitting = false;
app.on('before-quit', (event) => {
  if (isQuitting) return;
  isQuitting = true;
  event.preventDefault();
  stopBackend().then(() => app.quit());
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
