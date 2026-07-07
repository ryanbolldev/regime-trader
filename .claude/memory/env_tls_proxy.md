---
name: env_tls_proxy
description: This host's TLS-inspection proxy breaks all Python HTTPS unless routed through the OS trust store via enable_os_trust_store()
metadata:
  type: project
---

Ryan's Windows machine runs a TLS-inspection proxy (antivirus/corporate) whose root CA is installed in the **Windows trust store but not in certifi**. Consequently **every outbound HTTPS call from Python fails** with `SSL: CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` — this hits `requests`/`urllib3` (Alpaca SDK, alerts webhook) *and* `httpx` (tastytrade), not just one library.

**Fix:** `config.credentials.enable_os_trust_store()` (calls `truststore.inject_into_ssl()`, idempotent) routes TLS validation through the OS trust store, which *does* contain the proxy CA. One call patches the stdlib `ssl` module process-wide, fixing Alpaca + alerts + tastytrade at once. Must run **before the first HTTPS call**.

**Where it's wired:** `wheel_main.WheelTrader.startup()` (first line) and `wheel_scanner.options_data_tastytrade.build_session()`. Proven live: after adding it, the wheel_main dry-run reached "Account verified" against Alpaca (previously failed at `get_account()`).

**Latent gap — other entry points still lack it:** `main.py`, `scripts/connection_test.py` (Alpaca), and `scripts/run_scanner.py` do NOT call `enable_os_trust_store()`, so they'll hit the same SSL error on this host unless run in an environment without the proxy (e.g. Docker/CI). Add the one-liner at each entry point's startup if/when they're run here. `scripts/tastytrade_connection_test.py` and `scripts/tastytrade_parity_check.py` already call it. See [[task_tastytrade_integration]] and [[feedback_env_offlimits]].
