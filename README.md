# Denoise Studio

Professional photo denoising web app for Lightroom Mobile exports. Applies OpenCV Non-Local Means (NLM) denoising with per-channel luminance and color noise controls. Works as a PWA on iPad and desktop.

---

## Architecture

```
Browser (Next.js 14, Vercel)
    │
    │  POST /analyze      → JSON (smart crop zones + base64 thumbnails)
    │  POST /denoise      → binary image or JSON (preview crops)
    │  GET  /health       → JSON
    ▼
FastAPI + OpenCV (Railway, Docker)
    │
    └─ In-memory processing (stateless, no storage)
```

```
denoise-studio/
├── frontend/         Next.js 14 App Router → Vercel
│   ├── app/          Routes and global styles
│   ├── components/   UI components (UploadZone, PreviewCards, …)
│   ├── lib/          API client and utilities
│   └── public/       PWA manifest and icons
└── backend/          FastAPI + OpenCV → Railway
    ├── main.py       Endpoint definitions
    ├── denoise.py    NLM processing logic
    ├── analyze.py    Smart crop zone detection
    ├── schemas.py    Pydantic models
    ├── Dockerfile    Railway build
    └── railway.toml  Railway deploy config
```

---

## Local Development

### Backend

```bash
cd backend

# Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is available at http://localhost:8000.
Interactive docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install

# Copy env file and set backend URL
cp .env.example .env.local
# Edit .env.local if your backend runs on a different port

npm run dev
```

The app is available at http://localhost:3000.

---

## Environment Variables

### Frontend (Vercel)

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Yes | Full URL of the Railway backend (e.g. `https://denoise-studio.up.railway.app`) |

### Backend (Railway)

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALLOWED_ORIGIN` | No | `*` | CORS allowed origin. Set to your Vercel frontend URL in production (e.g. `https://denoise-studio.vercel.app`) |
| `WORKER_THREADS` | No | `2` | Number of threads in the NLM thread pool |
| `PORT` | Auto | `8000` | Set automatically by Railway — do not override |

---

## Deployment

### Backend → Railway

1. Push the repo to GitHub.
2. Create a new Railway project → **Deploy from GitHub repo**.
3. Point Railway at the `/backend` directory (set **Root Directory** to `backend`).
4. Railway auto-detects the `Dockerfile` and `railway.toml`.
5. Add the `ALLOWED_ORIGIN` environment variable in Railway dashboard → your Vercel URL.
6. Copy the Railway public URL (e.g. `https://denoise-studio.up.railway.app`).

### Frontend → Vercel

1. Create a new Vercel project → **Import from GitHub**.
2. Set **Framework Preset** to `Next.js`.
3. Set **Root Directory** to `frontend`.
4. Add environment variable `NEXT_PUBLIC_API_URL` → paste the Railway backend URL.
5. Deploy. Vercel auto-deploys on every push to `main`.

### PWA Icons

Before deploying, generate the PWA icons once:

```bash
cd frontend/public/icons
pip install Pillow
python generate_icons.py
```

Commit the generated `.png` files.

---

## Supported Formats

| Format | Input | Output |
|---|---|---|
| JPEG | ✅ | ✅ (default) |
| TIFF | ✅ | ✅ |
| AVIF | ✅ | — |
| JXL | ✅ | — |
| DNG / RAW / HEIC | ❌ — clear error shown | — |

Maximum input file size: **60 MB**.
Output is always at full original resolution (no downscaling).

---

## API Reference

### `GET /health`
Returns `{"status": "ok", "version": "1.0.0"}`.

### `POST /analyze`
**Body:** `multipart/form-data` with `file` (image).

**Returns:** JSON with `regions` (array of 3 zone objects), `image_width`, `image_height`.

Each zone: `{x, y, w, h, label, thumbnail_base64}` where `label` is one of `highlights | shadows | faces | center`.

### `POST /denoise`
**Body:** `multipart/form-data`

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | File | — | Image to denoise |
| `luminance_strength` | float 0–1 | `0.4` | Luminance noise strength |
| `color_strength` | float 0–1 | `0.3` | Color noise strength |
| `output_format` | `jpeg`\|`tiff` | `jpeg` | Output encoding |
| `preview_mode` | bool | `false` | If true, denoise only crop regions |
| `preview_regions` | JSON string | `[]` | Array of `{x,y,w,h}` when `preview_mode=true` |

**Returns (preview_mode=false):** Binary image with `Content-Disposition: attachment`.

**Returns (preview_mode=true):** JSON `{"previews": ["<base64>", …]}`.

### `POST /denoise-batch`
Accepts up to 10 files, returns a ZIP archive. Scaffolded — set `BATCH_ENABLED = true` in `frontend/app/page.tsx` to expose in UI.

---

## NLM Parameter Mapping

| Slider (0–100) | Backend parameter | Range |
|---|---|---|
| Luminance | `h` | 1–14 |
| Color | `hColor` | 1–10 |

Fixed: `templateWindowSize=7`, `searchWindowSize=21`.
