# Deployment

- **Backend** (FastAPI + C++/ONNX) → **Google Cloud Run** (Docker, scales to zero, ~seconds cold start)
- **Frontend** (Next.js) → **Vercel** (free, no spin-down) — or Cloud Run if you want one platform

---

## Backend → Google Cloud Run

### One-time setup
1. Install the gcloud SDK: https://cloud.google.com/sdk/docs/install
2. Sign in and pick a project:
   ```
   gcloud auth login
   gcloud projects create fake-news-123 --name="Fake News"   # or use an existing project
   gcloud config set project fake-news-123
   ```
3. **Enable billing** on the project (Console → Billing). Required even for the free tier — usage
   stays free within the monthly limits below; you just need a card on file.
4. Enable the APIs:
   ```
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
   ```

### Deploy (from the repo root)
```
gcloud run deploy fake-news-api \
  --source backend \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --concurrency 8 --port 8080
```
- `--source backend` builds `backend/Dockerfile` with Cloud Build (no manual `docker push`). Answer
  **yes** if prompted to create an Artifact Registry repo.
- Copy the **Service URL** it prints at the end, e.g. `https://fake-news-api-xxxxx.run.app`.
- Test it: `curl https://fake-news-api-xxxxx.run.app/` → `{"status":"ok"}`.

### Knobs
- `--min-instances 0` → scales to zero = **free**; cold start ~3–8s (vs Render's ~60s). Use
  `--min-instances 1` for always-warm, but that runs 24/7 and is **not** free.
- `--concurrency 8` — the backend reloads the model per request, so keep concurrency modest to avoid
  memory spikes. If you see OOM, raise `--memory 2Gi`.
- **Free tier:** 2M requests + 360k GB-sec + 180k vCPU-sec / month — plenty for a demo.

---

## Frontend → Vercel
1. https://vercel.com/new → import `NeeleshN1/NLP-Fake-News-Detector`.
2. **Root Directory:** `ui`  (Framework auto-detects as Next.js).
3. **Environment Variables:**
   | Name | Value |
   |---|---|
   | `NEWSAPI_KEY` | your NewsAPI key (server-side only) |
   | `MODEL_API_BASE` | `https://fake-news-api-xxxxx.run.app` (Cloud Run URL) |
   | `NEXT_PUBLIC_API_URL` | `https://fake-news-api-xxxxx.run.app` (client-side, Analyze sheet) |
4. Deploy. CORS already works — the FastAPI backend sends `Access-Control-Allow-Origin: *`.

### Want everything on GCP instead?
The Next.js app can also run on Cloud Run (add `output: "standalone"` to `next.config.ts` + a Node
Dockerfile). Vercel is simpler and free; ask if you'd prefer the all-Cloud-Run path.

---

## Note on prediction latency
The backend reloads the 67MB model on every request (per-request subprocess). Cold start is fixed by
Cloud Run, but for snappy *steady-state* predictions, switch to a warm long-lived process that loads
the model once. That's independent of hosting — happy to implement it.
