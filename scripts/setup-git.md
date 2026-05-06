# Git Setup

Run these commands once in the repo root (`C:\Users\Asus\OneDrive\Documents\Claude\Projects\Lumen`)
to initialize the git repo and make the initial commits. Use **PowerShell** or **Git Bash**
on Windows — OneDrive's file locking interferes with running git from inside the dev sandbox,
so this step has to be done on the host machine.

## 1. Clean up any stale `.git` directory

If a `.git/` folder already exists in the repo root, remove it first:

```powershell
# PowerShell
Remove-Item -Recurse -Force .git
```

```bash
# Git Bash / WSL
rm -rf .git
```

## 2. Initialize and configure

```bash
git init -b main
git config user.name "Ahmad Hamdan"
git config user.email "ahmadkhamdan3@gmail.com"
```

## 3. First commit (or commit history in chunks)

The simplest path is a single initial commit:

```bash
git add .
git commit -m "Initial Sprint 1 commit: monorepo, protocol, backend, frontend"
```

If you want a clean staged commit history (visible in `git log`) that mirrors the
Sprint 1 build order, see `scripts/git-commit-history.sh` for a script that creates
~10 commits, each scoped to one logical chunk of work.

## 4. (Optional) Push to GitHub

Create an empty repo on GitHub (don't initialize it with a README), then:

```bash
git remote add origin git@github.com:<your-username>/lumen.git
git push -u origin main
```

Or use HTTPS:

```bash
git remote add origin https://github.com/<your-username>/lumen.git
git push -u origin main
```
