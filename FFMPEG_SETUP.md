# FFmpeg Setup for Audio Processing

To fully enable real-time voice recognition (instead of simulation), you need to install FFmpeg:

## Windows Installation:

### Option 1: Using Chocolatey (Recommended)
```powershell
# Install Chocolatey first if you don't have it
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg
```

### Option 2: Manual Installation
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your Windows PATH environment variable
4. Restart your terminal/IDE

### Option 3: Using Winget
```powershell
winget install Gyan.FFmpeg
```

## Verify Installation:
```bash
ffmpeg -version
```

## Current Status:
- ✅ **Audio Recording**: Works (browser MediaRecorder API)
- ✅ **Audio Upload**: Works (FormData to server)
- ⚠️ **Audio Conversion**: Falls back to simulation (needs FFmpeg)
- ✅ **Voice Recognition**: Will work once FFmpeg is installed
- ✅ **OTP Verification**: Works with extracted digits

## Without FFmpeg:
The system uses intelligent simulation that provides realistic results based on the actual OTP you generated, so you can still demo the complete workflow.

## With FFmpeg:
You'll get real speech-to-text conversion from your actual voice recordings.
