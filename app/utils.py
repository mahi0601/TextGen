from transformers import AutoTokenizer
from app.config import MODEL_NAME, MAX_LENGTH

_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Ensure a valid pad token exists
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token

def tokenize(text: str):
    return _tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

def detokenize(ids):
    return _tokenizer.decode(ids, skip_special_tokens=True)

def batch_tokenize(texts: list):
    return _tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
