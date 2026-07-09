# SONIC Hosting Guide

SONIC runs from `webui.py`. The terminal CLI stays in `app.py`, so local chat and hosted Gradio use separate entrypoints.

## 1. Normal Local Run

PowerShell:

```powershell
python webui.py
```

Optional explicit environment values:

```powershell
$env:SONIC_HOST = "0.0.0.0"
$env:SONIC_PORT = "7860"
$env:SONIC_SHARE = "false"
python webui.py
```

## 2. Same-WiFi / LAN Run

LAN access is enabled by binding Gradio to `0.0.0.0`.

PowerShell:

```powershell
$env:SONIC_HOST = "0.0.0.0"
$env:SONIC_PORT = "7860"
$env:SONIC_SHARE = "false"
python webui.py
```

After startup, SONIC prints:

```text
SONIC launch settings
  Host: 0.0.0.0
  Port: 7860
  Share: off
  Local URL: http://127.0.0.1:7860
  LAN URL: http://192.168.x.x:7860
```

Use the LAN URL from another device on the same Wi-Fi.

## 3. Temporary Public Share Link

Use Gradio share mode only when you want a temporary public tunnel.

PowerShell:

```powershell
$env:SONIC_SHARE = "true"
python webui.py
```

Equivalent CLI:

```powershell
python webui.py --share
```

To force share mode off:

```powershell
python webui.py --no-share
```

## 4. Hugging Face Spaces

SONIC is prepared for Spaces through the root `README.md` front matter:

```yaml
sdk: gradio
app_file: webui.py
```

Deployment notes:

```text
1. Create a Gradio Space.
2. Push this repository.
3. Add secrets in the Space settings for any online-only features.
4. Use an external Ollama endpoint if you want local-mode behavior outside your own machine.
```

Important:

```text
Spaces is already public, so leave SONIC_SHARE off.
The app still writes chat history under data/chats, so persistent storage matters if you want history to survive restarts.
```

## 5. Render

Use this start command:

```powershell
python webui.py
```

Render usually provides a `PORT` environment variable. SONIC will use that automatically when `SONIC_PORT` is not set.

Recommended env vars:

```text
SONIC_HOST=0.0.0.0
SONIC_SHARE=false
SERPAPI_API_KEY=...
OLLAMA_URL=...
OLLAMA_MODEL=...
```

## 6. Railway

Use this start command:

```powershell
python webui.py
```

Railway commonly provides `PORT` automatically. SONIC will honor it when `SONIC_PORT` is not set.

Recommended env vars:

```text
SONIC_HOST=0.0.0.0
SONIC_SHARE=false
SERPAPI_API_KEY=...
OLLAMA_URL=...
OLLAMA_MODEL=...
```

## Supported Env Vars

```text
SONIC_HOST=0.0.0.0
SONIC_PORT=7860
SONIC_SHARE=false
SONIC_DEFAULT_RUNTIME=local
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3.2:3b
SERPAPI_API_KEY=
ENABLE_LIVE_WEB=true
```

Platform fallbacks:

```text
PORT is honored automatically on hosted platforms when SONIC_PORT is not set.
GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, and GRADIO_SHARE are also accepted for compatibility.
```

## Deployment Notes

```text
1. Local Ollama is required for SONIC local mode.
2. Hosted platforms usually do not include Ollama by default.
3. Online mode requires SERPAPI_API_KEY when ENABLE_LIVE_WEB is on.
4. Chat history and saved notes live under data/chats and need writable storage.
5. If the filesystem is ephemeral, history will reset after redeploys or restarts.
6. Host binding now defaults to 0.0.0.0, so other devices on the same network can reach the app.
```
