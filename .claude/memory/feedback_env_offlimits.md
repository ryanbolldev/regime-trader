---
name: feedback_env_offlimits
description: Never read or print the .env file or credential values; verify creds via connection tests instead
metadata:
  type: feedback
---

The `.env` file is strictly off-limits — do not read it, grep it, or print/echo any credential values (even lengths or masked forms). Ryan manages secrets himself and verifies their presence.

**Why:** Secrets must never enter the conversation transcript or tool output. Confirming presence by reading the file still exposes it and defeats the purpose.

**How to apply:** To validate credentials, write and run a **connection test** that exercises the API (e.g. OAuth handshake + a small data pull) and reports success/failure — prove accuracy by behavior, never by inspecting the secret. Credentials load exclusively through [[config_settings]]-style accessors in `config/credentials.py`. See [[task_tastytrade_integration]] for the tastytrade connection-test plan.
