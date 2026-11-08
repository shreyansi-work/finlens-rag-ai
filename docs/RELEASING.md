# Releasing (all free)

Pushing a version tag runs `.github/workflows/release.yml`, which fills the **Releases**, **Packages** and
**Deployments** sections on the repo's home page. It uses only the built-in `GITHUB_TOKEN`; GitHub Packages is
free for public packages and Actions is free for public repositories.

## One-time setup
1. Repo home page → gear icon next to **About** → under **Include in the home page** tick **Releases**,
   **Packages** and **Deployments** → **Save changes**.
2. Settings → Actions → General → **Workflow permissions**: leave on *Read and write permissions* (needed to
   create the release and push the image), or keep the default, since the workflow declares its own permissions.

## Publish a release
```powershell
cd E:\AI-Projects\finlens-rag-ai
git tag -a v0.1.0 -m "First public release"
git push origin v0.1.0
```
Watch it under the repo's **Actions** tab (workflow `release`). When it is green:
- **Releases** shows `v0.1.0` with generated notes.
- **Deployments** shows an entry for the `ghcr` environment.
- **Packages** shows `finlens-rag-ai` (the container image).

## Make the package public (important, once)
A new container package can start **private**, and private storage counts against your free quota.
1. Repo home → **Packages** → click `finlens-rag-ai` (or github.com → your profile → Packages).
2. **Package settings** → scroll to **Danger Zone** → **Change visibility** → **Public**.

## Pull and run the published image
```powershell
docker run -p 8000:7860 ghcr.io/shreyansi-work/finlens-rag-ai:latest
```

## Next release
Bump the version: `git tag -a v0.2.0 -m "..."` then `git push origin v0.2.0`.
