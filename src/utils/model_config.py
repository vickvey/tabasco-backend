# utils/model_config.py
from transformers import BertModel, BertTokenizer
from models.disamb_model import DisambModel
import torch

def get_disamb_model():
    bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    return DisambModel(bert_model, bert_tokenizer, device)