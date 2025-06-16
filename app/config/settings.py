import os
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
from typing import Annotated
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# ------------------------------------------------------------------------------
# PROJECT FOLDERS
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "static" / "uploads"
SUMMARY_FOLDER = PROJECT_ROOT / "static" / "summary"
DETAILED_FOLDER = PROJECT_ROOT / "static" / "detailed"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure necessary directories exist
for folder in (UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# ENVIRONMENT TYPE ENUM
# ------------------------------------------------------------------------------
class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"

# ------------------------------------------------------------------------------
# LOAD ENVIRONMENT FILE
# ------------------------------------------------------------------------------
env_str = os.getenv("ENVIRONMENT", "development")

try:
    environment = EnvironmentType(env_str.lower())
except ValueError:
    raise ValueError(f"❌ Invalid ENVIRONMENT value: {env_str}")

env_file = PROJECT_ROOT / f".env.{environment.value}.local"

if not env_file.is_file():
    raise FileNotFoundError(
        f"❌ Environment file not found: {env_file}. "
        f"Make sure the correct .env.{environment.value}.local file exists."
    )

load_dotenv(env_file)
print(f"✅ Loaded environment from: {env_file}")

# ------------------------------------------------------------------------------
# APP SETTINGS
# ------------------------------------------------------------------------------
class Settings(BaseSettings):
    # Basic
    PROJECT_NAME: str = "TABASCO FastAPI"
    RELEASE_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: Annotated[int, Field(ge=1024, le=65535)] = os.getenv("PORT", "8080")

    # App environment
    ENVIRONMENT: EnvironmentType = environment

    # Logging
    LOG_LEVEL: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_TO_FILE: bool = False
    LOG_FILE_PATH: str = str(LOG_DIR / "app.log")

    model_config = SettingsConfigDict(env_file_encoding="utf-8")

# ------------------------------------------------------------------------------
# SINGLETON SETTINGS INSTANCE
# ------------------------------------------------------------------------------
try:
    settings = Settings()
except ValidationError as e:
    print(f"❌ Settings validation error: {e}")
    raise
