const { app, BrowserWindow } = require('electron');
const path = require('path');
const { exec } = require('child_process');

let mainWindow;

// Add logging to debug Electron app
app.on('ready', () => {
    console.log('Electron app is starting...');

    // Start the Python web server
    const pythonServer = exec('source env/bin/activate && python src/ui/web_ui.py');

    pythonServer.stdout.on('data', (data) => {
        console.log(`Python: ${data}`);
    });

    pythonServer.stderr.on('data', (data) => {
        // Filter out debug logs from error output
        if (!data.toLowerCase().includes('debug')) {
            console.error(`Python Error: ${data}`);
        }
    });

    pythonServer.on('close', (code) => {
        console.log(`Python exited with code ${code}`);
    });

    // Create the Electron window
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
        },
    });

    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
        console.error(`Failed to load URL: ${errorDescription} (Code: ${errorCode})`);
    });

    // Add a delay before loading the URL to ensure the Python server is ready
    setTimeout(() => {
        mainWindow.loadURL('http://127.0.0.1:8000');
    }, 3000); // 3-second delay

    // Open DevTools for debugging
    mainWindow.webContents.openDevTools();

    // Add a listener for will-redirect to debug infinite redirects
    mainWindow.webContents.session.webRequest.onBeforeRedirect((details) => {
        console.log(`Redirect detected: ${details.url} -> ${details.redirectURL}`);
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
        pythonServer.kill(); // Stop the Python server when the app is closed
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
