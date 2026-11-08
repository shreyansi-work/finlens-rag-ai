# Deploy FinLens for free on Render

Hugging Face now marks **Docker Spaces as paid** for free accounts, so Render is the recommended free host.
Result: a public URL like `https://finlens-rag-ai.onrender.com` for your resume.

Free-tier behaviour (from Render's docs, https://render.com/docs/free; re-check it, plans change):
- The service **sleeps after 15 minutes without traffic**. The next visit takes about **1 minute** to wake up
  (Render shows a loading page). Open the link yourself a minute before sharing it.
- The filesystem is ephemeral. That is fine here: the sample corpus is generated and indexed while the image
  builds. Feedback and traces written at runtime are lost on restart.
- 750 free instance hours per month. Render may ask for a payment method at signup; you do not have to
  upgrade anything. Never choose a paid plan by accident: pick **Free** on the plan screen.

The hosted demo runs in **rule-based mode** (the Dockerfile sets `FINLENS_LLM_PROVIDER=none`), on the
synthetic corpus only.

## Steps

1. **Sign up** at https://render.com using **Sign in with GitHub** (easiest, it links your repos).
2. Dashboard → **New +** → **Web Service**.
3. **Connect the repo:** choose `shreyansi-work/finlens-rag-ai`. If it is not listed, click
   *Configure account* and grant Render access to that repository.
4. Fill in the form:
   - **Name:** `finlens-rag-ai`
   - **Language / Runtime:** **Docker** (Render detects the `Dockerfile` at the repo root)
   - **Branch:** `main`
   - **Instance type / plan:** **Free**
   - **Environment variables:** none needed (the Dockerfile already sets everything)
   - Leave Dockerfile path and Docker context at their defaults.
5. Click **Deploy Web Service**. Watch the **Logs**. The first build takes several minutes (it installs the
   packages, generates 18 PDFs and indexes them).
6. When the logs say the service is live, open the URL at the top of the page (`...onrender.com`).
7. **Test it** with: *Compare Nilgiri and Sahyadri operating margins for FY25 and compute the difference.*
   Expect 24.3% vs 17.1% and a 7.2% difference, with clickable citations.
8. **Link it:** GitHub repo → About (gear icon) → Website → paste the URL. Add a "Live demo" line to the README.

Every `git push` to `main` redeploys automatically.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "No open ports detected" | The container must listen on `$PORT`. The Dockerfile does (`--port ${PORT:-7860}`). If you built an old image, push the latest `main`. |
| Build fails on `pip install` | Read the build log. Versions are pinned in `requirements.txt` (Python 3.12 image). Paste the log to get it fixed. |
| First visit shows a loading page for ~1 min | Normal wake-up from sleep on the free plan. |
| Page loads but answers fail | Open the service **Logs** tab and look for a Python traceback. |

## If you do not want a hosted demo

A recruiter can still see the project through the README screenshots, the CI badge and `docker compose up`.
A 60-90 second screen recording (GIF or MP4 linked from the README) is a good no-hosting substitute.

## Safety notes

- Synthetic data only; nothing sensitive is exposed.
- There is no auth or rate limiting (see the README's Limitations). Do not point a public service at real or
  private filings without adding both.
- Do not add API keys: the project never calls a paid API.
