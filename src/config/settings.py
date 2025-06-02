import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
SUMMARY_FOLDER = PROJECT_ROOT / "static" / "summary"
DETAILED_FOLDER = PROJECT_ROOT / "static" / "detailed"
PORT = int(os.getenv("PORT", 8000))
ENV = os.getenv("ENV", "development")

# Load the environment file based on NODE_ENV
env_file = f".env.{os.getenv('NODE_ENV', 'development')}.local"
load_dotenv(env_file)

class Settings:
    PORT: int = int(os.getenv("PORT", 8000))  # Default port is 8000 if not set
    ENV: str = os.getenv("ENV", "development")