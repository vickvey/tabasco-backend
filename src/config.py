from pathlib import Path
from typing import Annotated
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Model Configs
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='forbid',
    )

    # App Constants and Configs
    PROJECT_NAME: str = "TABASCO FastAPI"
    RELEASE_VERSION: str = "1.0.0"
    ALLOWED_EXTENSIONS: set = {"pdf", "txt"}

    # Environment Variables
    ENVIRONMENT: str = Field(default="development")
    HOST: str = Field(default="0.0.0.0")
    PORT: Annotated[int, Field(ge=1024, le=65535)] = 8080

    # Project Folders
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER: Path = PROJECT_ROOT / "static" / "uploads"
    SUMMARY_FOLDER: Path = PROJECT_ROOT / "static" / "summary"
    DETAILED_FOLDER: Path = PROJECT_ROOT / "static" / "detailed"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    def __init__(self, **kwargs):
        # 👇 Check if the .env file exists before anything else
        env_path = Path(".env")
        if not env_path.is_file():
            raise FileNotFoundError("❌ Missing required .env file in project root")

        super().__init__(**kwargs)

        for folder in (self.UPLOAD_FOLDER, self.SUMMARY_FOLDER, self.DETAILED_FOLDER, self.LOG_DIR):
            folder.mkdir(parents=True, exist_ok=True)

# Singleton settings instance
try:
    settings = Settings()
except ValidationError as e:
    print(f"❌ Settings validation error: {e}")
    raise

