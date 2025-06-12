from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Annotated
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError

# ------------------------------------------------------------------------------
# PROJECT FOLDERS
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
SUMMARY_FOLDER = PROJECT_ROOT / "static" / "summary"
DETAILED_FOLDER = PROJECT_ROOT / "static" / "detailed"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
for folder in (UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# ENVIRONMENT SETTINGS
# ------------------------------------------------------------------------------
# Load environment file at module level
env_file = PROJECT_ROOT / f".env.{os.getenv('RUN_ENV', 'development')}.local"
if not env_file.is_file():
    env_file = PROJECT_ROOT / ".env.local"

if env_file.is_file():
    load_dotenv(env_file)
    print(f"✅ Loaded environment from: {env_file}")
else:
    print(f"⚠️ No environment file found at {env_file}. Using defaults.")

class Settings(BaseSettings):
    PORT: Annotated[int, Field(ge=1024, le=65535)] = 8000
    RUN_ENV: str = "development"
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: str = str(LOG_DIR / "app.log")

    model_config = SettingsConfigDict(env_file_encoding="utf-8")

# Instantiate settings as a singleton
try:
    settings = Settings()
except ValidationError as e:
    print(f"❌ Settings validation error: {e}")
    raise