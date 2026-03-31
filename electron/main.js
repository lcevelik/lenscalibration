const { app, BrowserWindow, ipcMain } = require('electron');
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

    const onReady = (data) => {
      if (data.toString().includes('Application startup complete')) {
        resolve();
      }
    };

    backendProcess.stdout.on('data', onReady);
    backendProcess.stderr.on('data', onReady);
    backendProcess.on('error', reject);

    // Fallback: resolve after 4s if the startup log never appears
    setTimeout(resolve, 4000);
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

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
