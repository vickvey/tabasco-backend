# app/types.py
from app.settings import settings
from app.models.disamb_model import DisambModel

# Define AppState type
class AppState:
    settings: 'settings.Settings'
    disamb_model: DisambModel