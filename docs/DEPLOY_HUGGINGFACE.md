# Deploy FinLens to Hugging Face Spaces

Result: a public URL like `https://<your-hf-username>-finlens.hf.space` that you can put on your resume.
Free CPU tier, no GPU, no Ollama, so the hosted demo runs in **rule-based mode** (the Dockerfile sets
`FINLENS_LLM_PROVIDER=none`). The sample corpus is generated and indexed while the image builds.

## Why there is an export script

A Space repo needs a `README.md` with a YAML header, and Hugging Face **rejects pushes that contain binary
files** (PNG screenshots, PDFs, databases) unless they use Git LFS. `scripts/export_space.py` copies only what
the Dockerfile needs and writes a short Space README, so the push is clean.

## Steps

1. **Account.** Sign up at https://huggingface.co (free) and verify your email.
2. **Access token.** Settings → Access Tokens → *New token* → type **Write** → copy it (you will paste it as the
   password when `git` asks). Never commit it or paste it in the README.
3. **Create the Space.** https://huggingface.co/new-space
   - Space name: `finlens`
   - License: leave empty (optional)
   - SDK: **Docker** → template **Blank**
   - Hardware: **CPU basic (free)**
   - Visibility: **Public**
4. **Clone the empty Space** next to your project (replace `<user>`):
   ```bash
   git clone https://huggingface.co/spaces/<user>/finlens ../finlens-space
   ```
   Username = your HF username, password = the token from step 2.
5. **Export the project into it** (from the `finlens-rag-ai` folder):
   ```bash
   python scripts/export_space.py ../finlens-space --github https://github.com/shreyansi-work/finlens-rag-ai
   ```
   It prints the number of files and warns if any binary files slipped in (there should be none).
6. **Push:**
   ```bash
   cd ../finlens-space
   git add .
   git commit -m "Deploy FinLens demo"
   git push
   ```
7. **Watch the build.** Open the Space → **Logs** tab. The first build takes a few minutes (installing packages,
   generating 18 PDFs, ingesting). When the status shows **Running**, open the **App** tab.
8. **Test it** with: *Compare Nilgiri and Sahyadri operating margins for FY25 and compute the difference.*
   You should see 24.3% vs 17.1%, a 7.2% difference and clickable citations.
9. **Link it** from the GitHub README (a "Live demo" line near the top) and from your resume.

## Updating the demo later

Re-run step 5 (it overwrites files), then `git add . && git commit -m "Update" && git push` inside `finlens-space`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Push rejected: "binary files" | Something non-text was copied. Check the export script's WARNING line; remove those files and amend. |
| Build error `pip install` | Read the Logs tab. Versions are pinned in `requirements.txt`; the Dockerfile uses Python 3.12. |
| Space says "Runtime error", port issues | The Dockerfile must listen on **7860** and the README header must say `app_port: 7860` (both already set). |
| App is slow on first load | Free Spaces sleep after inactivity; the first request wakes them (about 30-60 s). |
| Feedback or traces disappear | The container filesystem is ephemeral. Fine for a demo; use a database or a Space volume for persistence. |

## Safety notes for a public demo

- It serves **synthetic data only**, so nothing sensitive is exposed.
- There is no auth or rate limiting (see the README's Limitations). Do not point a public Space at real or
  private filings without adding both.
- Do not add an API key to the Space: the project never calls a paid API.
