const { spawn } = require('child_process');
const electronBin = require('electron');
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE;
const child = spawn(electronBin, ['.'], { stdio: 'inherit', env });
child.on('close', (code) => process.exit(code ?? 1));
