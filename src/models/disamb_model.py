import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import torch

class DisambModel:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def forward(self, input_sentence, target_word):
        marked_text = "[CLS] " + input_sentence + " [SEP]"
        tokens = self.tokenizer.tokenize(marked_text)
        target_tokens = self.tokenizer.tokenize(target_word.lower())

        # Find the target word's subword indices
        for i in range(len(tokens) - len(target_tokens) + 1):
            if tokens[i:i + len(target_tokens)] == target_tokens:
                target_start = i
                target_end = i + len(target_tokens) - 1
                break
        else:
            raise ValueError(f"Target word '{target_word}' not found in sentence.")

        # Tokenize and get embeddings
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensor = torch.tensor([segments_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensor)
            hidden_states = outputs.hidden_states

        # Average embeddings of target word subwords
        target_embs = []
        for j in range(target_start, target_end + 1):
            layers = [hidden_states[-k][0][j] for k in range(1, 5)]
            token_emb = torch.sum(torch.stack(layers), dim=0)
            target_embs.append(token_emb)
        target_emb = torch.mean(torch.stack(target_embs), dim=0)
        return target_emb

    def get_context_words(self, sentence, target_word, top_k=10):
        tokens, embeddings = self.forward(sentence)
        target_word = target_word.lower()

        # Get positions of the target word in the token list
        target_indices = [i for i, token in enumerate(tokens) if target_word in token]
        if not target_indices:
            return [], []

        target_vector = embeddings[target_indices[0]]

        similarities = []
        for i, (token, vector) in enumerate(zip(tokens, embeddings)):
            if i == target_indices[0] or token in ["[CLS]", "[SEP]"]:
                continue
            similarity = F.cosine_similarity(target_vector, vector, dim=0).item()
            similarities.append((token, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        top_similar = similarities[:top_k]
        return top_similar