# Publish FinLens on GitHub (Windows, step by step)

## 0. Placeholders to fill in first
- `README.md`: replace `<YOUR_GITHUB_USERNAME>` (CI badge URLs) after you create the repo.

## 1. Install and configure Git
```powershell
winget install -e --id Git.Git
```
Open a **new** terminal, then:
```powershell
git --version
git config --global user.name "Your Name"
git config --global user.email "your-github-noreply-or-real-email"
```
(GitHub → Settings → Emails has a private `...@users.noreply.github.com` address if you do not want your real one in commits.)

## 2. Clean up before the first commit
```powershell
cd E:\AI-Projects\finlens-rag-ai
Rename-Item .venv .venv-mac-old        # the old macOS venv; it is git-ignored. Delete it if you prefer.
python -m pytest -q                    # expect all tests to pass
python -m ruff check .                 # expect "All checks passed"
```

## 3. Create the local repository
```powershell
git init -b main
git add .
git status                             # READ THIS: no .venv, .DS_Store, data/*.db, data/*.npz, .env, .serena
git ls-files | Measure-Object          # sanity check on the file count
```
If anything unwanted is listed, add it to `.gitignore`, then `git rm -r --cached <path>` and `git add .` again.
**Never commit** `.env`, tokens or API keys (this project needs none).

## 4. Make clean, meaningful commits
A readable history is part of the résumé. Reset the staging area and commit in logical pieces:
```powershell
git reset
git add .gitignore requirements.txt requirements-dev.txt pyproject.toml
git commit -m "chore: project scaffolding, pinned dependencies and lint config"

git add finlens scripts web
git commit -m "feat: agentic RAG pipeline (ingestion, hybrid retrieval, agent tools, API, UI)"

git add data/raw eval
git commit -m "data: synthetic filings corpus, golden set and hand-written held-out set"

git add tests
git commit -m "test: unit, end-to-end and API tests with CI smoke eval"

git add Dockerfile .dockerignore docker-compose.yml .github
git commit -m "ci: lint + Linux/Windows test matrix; Docker and Compose"

git add README.md docs
git commit -m "docs: README with ablation, held-out results, limitations and deploy guides"

git status                             # should be clean
git log --oneline
```

## 5. Create the GitHub repository
1. https://github.com/new
2. Repository name: `finlens-rag-ai` · **Public**
3. **Do not** tick "Add a README", or ".gitignore" (you already have them), and choose no license.
4. Click **Create repository**.

## 6. Push
```powershell
git remote add origin https://github.com/<your-username>/finlens-rag-ai.git
git push -u origin main
```
A browser window (Git Credential Manager) will ask you to sign in; approve it. If you are asked for a password
in the terminal, use a **personal access token** (GitHub → Settings → Developer settings → Tokens), not your
account password.

## 7. After the push
1. **Actions tab:** confirm the `ci` workflow turns green on both Ubuntu and Windows. The workflow was written
   but not run before your first push; if it fails, open the failing step and send me the log.
2. Replace `<YOUR_GITHUB_USERNAME>` in `README.md`, then:
   ```powershell
   git add README.md; git commit -m "docs: fix CI badge URL"; git push
   ```
3. Repo **About** (gear icon): add a one-line description, the Hugging Face demo URL as the website, and topics
   such as `rag`, `llm`, `fastapi`, `retrieval-augmented-generation`, `sqlite`, `ollama`, `evaluation`.
4. Pin the repo on your GitHub profile.
5. Do the Hugging Face deploy next: [DEPLOY_HUGGINGFACE.md](DEPLOY_HUGGINGFACE.md).

## 8. Day-to-day
```powershell
git checkout -b feat/short-name        # work on a branch
git add -p; git commit -m "feat: ..."  # small, descriptive commits
git push -u origin feat/short-name     # open a pull request; CI runs on it
```
