from pathlib import Path
from dotenv import load_dotenv
import os

from pydantic import Field
from pydantic_settings import BaseSettings


# ------------------------------------------------------------------------------
# PROJECT FOLDERS
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
SUMMARY_FOLDER = PROJECT_ROOT / "static" / "summary"
DETAILED_FOLDER = PROJECT_ROOT / "static" / "detailed"

for folder in (UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# ENVIRONMENT SETTINGS
# ------------------------------------------------------------------------------
class Settings(BaseSettings):
    PORT: int = Field(default=8000, description="Port to listen on", env="PORT")
    RUN_ENV: str = Field(default="development", env="RUN_ENV")

    class Config:
        env_file_encoding = "utf-8"

# Load environment file dynamically
env_file = PROJECT_ROOT / f".env.{os.getenv('RUN_ENV', 'development')}.local"
if not env_file.is_file():
    env_file = PROJECT_ROOT / ".env.local"

if env_file.is_file():
    load_dotenv(env_file)
    print(f"✅ Loaded environment from: {env_file}")
else:
    print("⚠️  No environment file found. Skipping load_dotenv().")

