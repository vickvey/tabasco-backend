import torch
from fastapi import Depends, Request
from transformers import BertTokenizer, BertModel
from typing import Annotated
from src.models import DisambModel

# Getter functions
def get_bert_model():
    return BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

def get_bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


# Dependency Declrations
bert_model_dependency = Annotated[BertModel, Depends(get_bert_model)]
bert_tokenizer_dependency = Annotated[BertTokenizer, Depends(get_bert_tokenizer)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_disamb_model(request: Request):
    return request.app.state.disamb_model