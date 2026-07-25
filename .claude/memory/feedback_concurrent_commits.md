---
name: feedback_concurrent_commits
description: Ryan sometimes commits manually mid-session while Claude is working — don't treat surprise commits under his identity as anomalies
metadata:
  type: feedback
---

Ryan sometimes runs `git commit` himself, in a separate terminal/IDE, while a Claude Code session is actively working in the same repo. Commits can appear under his git identity (`ryanbolldev`) that Claude did not create — e.g. a commit titled "setting up cron jobs" appeared mid-session containing Claude's in-progress `docker-compose.yml`/`wheel_main.py` edits verbatim, never committed by Claude itself.

**Why:** Ryan said this directly — "I do commits from time to time while you're working since we've had problems in the past with you committing things." This is a safeguard he adopted because of prior incidents with Claude's commit behavior, not a one-off.

**How to apply:**
- If a commit appears that wasn't made by this session, don't report it as a mystery or a bug — check its diff against what's expected (it's likely Ryan capturing in-progress work) and move on.
- This raises the bar on the project's existing git discipline rule (see CLAUDE.md: only commit when explicitly asked, never `--amend`, new commits only): be conservative about *when* to commit, and always show/summarize the diff being committed rather than committing silently, since commit hygiene is a known sore point for this user.
