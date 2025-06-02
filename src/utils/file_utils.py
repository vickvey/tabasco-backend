# utils/file_utils.py
from config.settings import UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER

def setup_directories():
    for folder in (UPLOAD_FOLDER, SUMMARY_FOLDER, DETAILED_FOLDER):
        folder.mkdir(parents=True, exist_ok=True)