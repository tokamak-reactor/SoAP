# SoAP — Backup and GitHub Push Workflow

Session date: July 30, 2026

## Scenario

User wanted to freeze and push the SoAP project to GitHub before potential computer issues.
The repo had no remote configured and had never been pushed.

## Steps performed

### 1. Check repo state

```bash
cd /home/kirill/projects/solps_analysis
git status          # → clean working tree
git remote -v       # → no remote configured
git branch -a       # → only master
```

### 2. Install `gh` CLI (Fedora)

```bash
sudo dnf install -y gh
```

### 3. GitHub authentication

Two approaches attempted:

**Approach A: PAT (Personal Access Token)**
- Token needs scopes: `repo` + `read:org`
- Pass via stdin: `echo "<token>" | gh auth login --with-token`
- First token was missing `read:org`, second was HTTP 401 (invalid)
- Fall back to SSH

**Approach B: SSH key**
```bash
ssh-keygen -t ed25519 -C "email@example.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
# → add to https://github.com/settings/keys
```

### 4. Git config (if not set)

```bash
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
```

### 5. Create GitHub repo and push

```bash
# After SSH key added to GitHub:
gh repo create solps_analysis --private --source=. --remote=origin --push
# Or manually:
git remote add origin git@github.com:<user>/solps_analysis.git
git push -u origin master
```

### 6. Save Hermes memory + skill into repo

Create `.hermes/` directory in the repo root with:
- `.hermes/memory-solps.md` — snapshot of relevant Hermes memory entries
- `.hermes/skill-solps-analysis-project.md` — snapshot of the skill content

Then commit and push again:
```bash
git add .hermes/ RECOVERY.md
git commit -m "backup: Hermes memory + skill snapshots"
git push
```

### 7. Create RECOVERY.md

Place in repo root with:
- Clone URL
- Required data paths (external watch directories)
- Hermes skill installation: `skill_manage(action='create', name='solps-analysis-project', ...)`
- Environment setup (virtualenv, pip install -e .)

## Pitfalls

- `gh auth login` requires `read:org` scope on PAT tokens in addition to `repo`
- If PAT auth fails repeatedly, SSH is the simpler and more reliable fallback
- Do NOT commit tokens or credentials to the repo — only the public SSH key
- The `.hermes/` directory in the repo is a *snapshot*, not a live sync — it must be re-created manually on the new machine
- Memory entries are limited to 2200 chars and may need consolidation before saving
