# Airacare Render Backend Deployment

Use this to host the YOLO backend for the Firebase web app.

## Backend Host

Recommended free demo host:

```text
Render Free Web Service
```

## Files Render Needs

```text
web_backend.py
src/
config/
models/airacare_animal_detector_best.pt
requirements-web.txt
render.yaml
```

## Render Settings

If creating manually:

```text
Environment: Python
Build command: pip install -r requirements-web.txt
Start command: gunicorn web_backend:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180
Plan: Free
```

## Health Check

After deployment, open:

```text
https://YOUR-RENDER-SERVICE.onrender.com/api/health
```

Expected:

```json
{
  "ok": true,
  "targetClasses": ["person", "dog", "cat", "horse", "cow", "deer", "goat"],
  "usingPretrainedFallback": false
}
```

## Final Web Link

Use the Render backend URL as the `api` query parameter:

```text
https://airacare-animal-safety.web.app/?api=https://YOUR-RENDER-SERVICE.onrender.com
```

That is the link to share after the Render backend health check works.
