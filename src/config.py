from pathlib import Path
from pydantic import ValidationError
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Constants and Configs
    PROJECT_NAME: str = "TABASCO FastAPI"
    RELEASE_VERSION: str = "1.0.0"
    ALLOWED_EXTENSIONS: set = {"pdf", "txt"}

    # Project Folders
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    ARTIFACTS_FOLDER: Path = PROJECT_ROOT / "artifacts"
    SESSION_FOLDER: Path = ARTIFACTS_FOLDER / "sessions"
    UPLOAD_FOLDER: Path = PROJECT_ROOT / "artifacts" / "uploads"
    LOG_DIR: Path = PROJECT_ROOT / "logs"
    # SUMMARY_FOLDER: Path = PROJECT_ROOT / "static" / "summary"
    # DETAILED_FOLDER: Path = PROJECT_ROOT / "static" / "detailed"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for folder in (
            self.ARTIFACTS_FOLDER,
            self.SESSION_FOLDER,
            self.LOG_DIR,
            self.UPLOAD_FOLDER
            # self.SUMMARY_FOLDER, 
            # self.DETAILED_FOLDER, 
            ):
            folder.mkdir(parents=True, exist_ok=True)

# Singleton settings instance
try:
    settings = Settings()
except ValidationError as e:
    print(f"❌ Settings validation error: {e}")
    raise

