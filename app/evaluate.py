from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

rouge = Rouge()

def evaluate_bleu(reference: str, generated: str):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    return sentence_bleu(reference_tokens, generated_tokens)

def evaluate_rouge(reference: str, generated: str):
    scores = rouge.get_scores(generated, reference, avg=True)
    return scores

def evaluate_all(reference: str, generated: str):
    return {
        "BLEU": evaluate_bleu(reference, generated),
        "ROUGE": evaluate_rouge(reference, generated)
    }
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

rouge = Rouge()

def evaluate_bleu(reference: str, generated: str):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    return sentence_bleu(reference_tokens, generated_tokens)

def evaluate_rouge(reference: str, generated: str):
    scores = rouge.get_scores(generated, reference, avg=True)
    return scores

def evaluate_all(reference: str, generated: str):
    return {
        "BLEU": evaluate_bleu(reference, generated),
        "ROUGE": evaluate_rouge(reference, generated)
    }
