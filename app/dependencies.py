from typing import Annotated
from fastapi import Depends

from app.services.sentence_processing import TextProcessingService


def get_text_processor():
    return TextProcessingService()


text_processor_dependency = Annotated[TextProcessingService, Depends(get_text_processor)]
