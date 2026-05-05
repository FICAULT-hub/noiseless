# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Noiseless** — a PWA photo denoising tool. Users upload a Lightroom Mobile export, tune two sliders (luminance + color noise), preview smart crops in real time, then download the full denoised file. Stateless: no auth, no storage, files processed entirely in memory.

## Monorepo layout

```
/backend    FastAPI + OpenCV → Railway (Docker)
/frontend   Next.js 15 PWA  → Vercel
```

## Local development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
# .env.local already present with NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev          # http://localhost:3000
npm run build        # production build (also validates types)
npm run type-check   # tsc --noEmit only
```

## Deployment

| Service | Platform | Trigger |
|---|---|---|
| Backend | Railway (Dockerfile) | push to `master` |
| Frontend | Vercel (Next.js) | push to `master` |

- Railway reads `backend/railway.toml`. Builder must be `"DOCKERFILE"` (uppercase). `restartPolicyType` must be `"ON_FAILURE"` (uppercase). No `startCommand` — the Dockerfile CMD handles `$PORT` via `sh -c "uvicorn ... --port ${PORT:-8000}"`.
- Vercel root directory is set to `frontend`. PWA service worker files (`sw.js`, `workbox-*.js`) are generated at build time and gitignored.

### Environment variables

**Railway** (set via dashboard or `railway variables set`):
- `ALLOWED_ORIGIN` — Vercel/custom domain URL for CORS (e.g. `https://noiseless.borborygmos-studio.ch`)

**Vercel** (set via dashboard or `vercel env add`):
- `NEXT_PUBLIC_API_URL` — Railway backend URL

### Custom domain
- Frontend: `noiseless.borborygmos-studio.ch` (CNAME → `cname.vercel-dns.com`)
- Backend: `https://noiseless-backend-production.up.railway.app`

## Architecture

### Backend (`backend/`)

Request flow: `main.py` → `analyze.py` or `denoise.py` → response

- **`main.py`** — FastAPI app, CORS, thread pool (`ThreadPoolExecutor`), file validation (60 MB limit, format allowlist), error handling. All heavy processing runs via `loop.run_in_executor` to avoid blocking the event loop.
- **`analyze.py`** — Detects 3 smart crop zones using LAB colorspace grid analysis (highlights, shadows) and OpenCV Haar cascade (faces, falls back to center). Returns base64 JPEG thumbnails.
- **`denoise.py`** — Wraps `cv2.fastNlMeansDenoisingColored`. Slider values (0–1) are lerped to NLM params: `h` = 1–14, `hColor` = 1–10, `templateWindowSize=7`, `searchWindowSize=21`. Preview mode denoises only the requested crop regions.
- **`schemas.py`** — Pydantic v2 models for all I/O.

**Known constraint:** NLM on full-resolution images is memory-intensive (2–4 GB for 24+ MP). Railway service must be set to ≥2 GB RAM. The `Killed` signal in logs = OOM.

### Frontend (`frontend/`)

Single page (`app/page.tsx`) with a 3-step state machine: `upload → configure → result`.

- **State flow:** file drop → POST `/analyze` → show `PreviewCards` + `DenoiseSliders` → sliders debounced 400ms → POST `/denoise` (preview_mode=true, per-crop) → "Denoise Full Image" → POST `/denoise` (full) → download + show `BeforeAfterViewer`.
- **`lib/api.ts`** — All fetch calls. `fetchPreviews` accepts an `AbortSignal` so in-flight preview requests are cancelled when sliders change before the response arrives.
- **`components/Loupe.tsx`** — `LoupeController` wraps any container and tracks pointer/touch. `Loupe` renders a 200px circular magnifier (3×) split left=original / right=denoised using CSS `background-image` + `clipPath`.
- **`components/BeforeAfterViewer.tsx`** — Drag-to-reveal split slider. Uses `ResizeObserver` to track container width so the clipped original image fills the full width correctly.
- **`components/ui/ToastProvider.tsx`** — Custom toast system built on Radix `@radix-ui/react-toast`. Use `useToast()` hook anywhere in the tree.

### Feature flags

`BATCH_ENABLED = false` in `app/page.tsx` — batch mode backend endpoint (`POST /denoise-batch`) is fully implemented but the UI is hidden. Flip to `true` to expose.

### Supported formats

Input: JPEG, TIFF, AVIF (via `pillow-avif-plugin`), JXL (via `imagecodecs` — removed from requirements due to PyPI yanking; gracefully absent).  
Output: JPEG or TIFF (user selectable).  
Rejected with structured error: DNG, RAW, HEIC.

## CLI tools available

```bash
# GitHub
"C:\Program Files\GitHub CLI\gh.exe" ...   # gh not on Bash PATH, use full path or PowerShell

# Railway (on Bash PATH via npm global)
railway deployment list
railway logs --build <deployment-id>
railway variables set KEY="value"

# Vercel (on Bash PATH via npm global)
vercel domains add <domain>
vercel env add <VAR> production
```
