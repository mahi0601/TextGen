from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def tokenize_text(text, tokenizer):
    return tokenizer.encode(text, return_tensors='pt')

def evaluate_output(reference, hypothesis):
    rouge = Rouge()
    bleu_score = sentence_bleu([reference.split()], hypothesis.split())
    rouge_scores = rouge.get_scores(hypothesis, reference)
    return {"BLEU": bleu_score, "ROUGE": rouge_scores}
