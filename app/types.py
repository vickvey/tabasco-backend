# app/types.py
from app.settings import Settings
from app.models.disamb_model import DisambModel

class AppState:
    settings: Settings
    disamb_model: DisambModel
