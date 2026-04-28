"""
config/credentials.py
---------------------
Sole point of entry for all credentials and secrets.

Loads .env on first call to load_credentials(), exposes typed accessors, and
fails fast if any required variable is absent.  Never logs, prints, or passes
credential values to any external service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when required credentials are missing or invalid."""


_REQUIRED_VARS = ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL")


@dataclass(frozen=True)
class Credentials:
    api_key:    str
    api_secret: str
    base_url:   str   # stripped of trailing slash

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url.lower()


def load_credentials() -> Credentials:
    """Load and validate credentials from the .env file.

    Raises ConfigurationError if any required variable is absent or if
    ALPACA_BASE_URL does not start with https://.
    """
    _load_dotenv_once()

    missing = [k for k in _REQUIRED_VARS if not os.getenv(k)]
    if missing:
        raise ConfigurationError(
            f"Missing required environment variables: {missing}. "
            "Copy .env.example to .env and fill in the values."
        )

    base_url = os.environ["ALPACA_BASE_URL"].rstrip("/")
    if not base_url.startswith("https://"):
        raise ConfigurationError(
            f"ALPACA_BASE_URL must use HTTPS (got: {base_url[:30]}…). "
            "Plain HTTP is not permitted."
        )

    return Credentials(
        api_key=os.environ["ALPACA_API_KEY"],
        api_secret=os.environ["ALPACA_API_SECRET"],
        base_url=base_url,
    )


def _load_dotenv_once() -> None:
    """Load .env into os.environ without overriding already-set variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return   # dotenv not installed; rely on real env vars (CI, containers)

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
