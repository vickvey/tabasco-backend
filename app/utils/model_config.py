from transformers import BertModel, BertTokenizer
from app.models.disamb_model import DisambModel
import torch
from fastapi import Request


def get_disamb_model(request: Request = None):
    # If request is provided and app state has the model, use it
    if request and hasattr(request.app.state, "disamb_model"):
        return request.app.state.disamb_model

    # Otherwise create a new model
    bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    return DisambModel(bert_model, bert_tokenizer, device)
