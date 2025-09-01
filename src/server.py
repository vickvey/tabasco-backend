import torch 
from functools import lru_cache
from transformers import BertTokenizer, BertModel
from fastapi import FastAPI
from contextlib import asynccontextmanager
from nltk.corpus import stopwords, wordnet as wn
from .utils import (
    ensure_nltk_data
)
from .models import DisambModel
from .routers import router as api_router
from .utils.logger import logger


@lru_cache()
def get_bert_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained("bert-base-uncased")

@lru_cache()
def get_bert_model() -> BertModel:
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device) # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download NLTK Resources once
    ensure_nltk_data()

    # Load stopwords once
    app.state.stop_words = set(stopwords.words("english"))

    # Build ALL_NOUNS set once
    all_nouns = set()
    for synset in wn.all_synsets('n'):
        for lemma in synset.lemmas():
            all_nouns.add(lemma.name().lower())
    app.state.all_nouns = all_nouns

    # Load BERT components from cached functions
    tokenizer = get_bert_tokenizer()
    model = get_bert_model()
    device = next(model.parameters()).device  # infer device from model

    # Set to app state
    app.state.bert_tokenizer = tokenizer
    app.state.bert_model = model
    app.state.disamb_model = DisambModel(model, tokenizer, device)

    yield

def init_routers(app: FastAPI):
    app.include_router(api_router, prefix='/api')

def create_app() -> FastAPI:
    logger.info("Starting Modern TABASCO backend")
    app = FastAPI(
        # Basic
        title="TABASCO FastAPI",
        # summary="A FastAPI REST API for detecting intra-domain ambiguities",
        description="A FastAPI REST API for detecting intra-domain ambiguities",
        version="1.0.0",
        lifespan=lifespan
    )

    @app.get("/health")
    def _health_check():
        return {"status": "ok"}

    init_routers(app=app)
    return app

# App singleton instance
app = create_app()
