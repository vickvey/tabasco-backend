# app/settings.py

from dotenv import load_dotenv
from pathlib import Path
import os
from typing import Optional

from pydantic_settings import BaseSettings as _BaseSettings
from pydantic import Field


# ------------------------------------------------------------------------------
# 1) PROJECT ROOT & FOLDER CONSTANTS
# ------------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER: Path = PROJECT_ROOT / "uploads"
SUMMARY_FOLDER: Path = PROJECT_ROOT / "static" / "summary"
DETAILED_FOLDER: Path = PROJECT_ROOT / "static" / "detailed"


def ensure_directories_exist() -> None:
    """
    Create UPLOAD_FOLDER, SUMMARY_FOLDER, and DETAILED_FOLDER if they don't already exist.
    Prints out their paths (for debugging).
    """
    for folder in (UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)

    # Debug prints for directory confirmation
    print("📁 PROJECT_ROOT:   ", PROJECT_ROOT)
    print("📁 UPLOAD_FOLDER:  ", UPLOAD_FOLDER)
    print("📁 SUMMARY_FOLDER: ", SUMMARY_FOLDER)
    print("📁 DETAILED_FOLDER:", DETAILED_FOLDER)


# ------------------------------------------------------------------------------
# 2) SETTINGS CLASS
# ------------------------------------------------------------------------------

class Settings(_BaseSettings):
    """
    Application settings, loaded from `.env.<environment>.local`.
    Environment is read from the `RUN_ENV` environment variable
    (defaults to "development").

    Example usage:
        >>> settings = Settings()
        >>> print (settings.RUN_ENV)
    """
    # Which environment to load: "development", "staging", "production", etc.
    RUN_ENV: str = Field(default="development", env="RUN_ENV")

    # Add other environment‐driven configuration here, for example,
    #   - API_KEY: Optional[str] = Field (None, env="API_KEY")
    #   - DEBUG_MODE: bool = Field(False, env="DEBUG_MODE")
    #
    # For now, we only load RUN_ENV.

    class Config:
        # We will override this __init__ to load the appropriate .env file before BaseSettings does its work.
        env_file: Optional[str] = None  # Will be set in __init__
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        """
        1. Ensure required directories exist.
        2. Determine which .env file to load based on RUN_ENV.
        3. Delegate to Pydantic to load environment variables.
        """
        # Step 1: Create folders if necessary
        ensure_directories_exist()

        # Step 2: Determine .env file name
        run_env = os.getenv("RUN_ENV", "development")
        env_filename = f".env.{run_env}.local"
        env_path = PROJECT_ROOT / env_filename

        # If that specific file doesn’t exist, fallback to ".env.local" or skip quietly
        if not env_path.is_file():
            fallback = PROJECT_ROOT / ".env.local"
            env_path = fallback if fallback.is_file() else None

        if env_path:
            load_dotenv(env_path)
            # Tell Pydantic which file was loaded
            object.__setattr__(self, "Config", type("C", (), {"env_file": str(env_path)}))
        else:
            print(f"⚠️  No environment file found for RUN_ENV='{run_env}'. Skipping load_dotenv().")

        # Finally, let Pydantic load any variables from the environment into this model
        super().__init__(**kwargs)


# ------------------------------------------------------------------------------
# 3) USAGE
# ------------------------------------------------------------------------------
# If you import Settings anywhere else, the above __init__ logic will run:
#   settings = Settings()
#
#  - Folders will already be created.
#  - The correct .env file is loaded (or skipped, with a warning).
#  - You can now reference settings.RUN_ENV (or other env_vars you add later).
#
# For backward‐compatibility, you could also do:
#   settings = Settings()
# without changing any code elsewhere.

