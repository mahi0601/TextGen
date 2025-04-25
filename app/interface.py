import torch
from app.text_generator import TextGenerator
from app.config import HIDDEN_DIM, NUM_LAYERS, MODEL_NAME, MODEL_PATH
from app.utils import tokenize, detokenize
from app.evaluate import evaluate_all

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextGenerator(pre_trained_model=MODEL_NAME).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
def predict(prompt: str):
    inputs = tokenize(prompt)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        predicted = torch.argmax(logits, dim=-1)
        generated = detokenize(predicted[0])
    return generated


def predict_and_evaluate(prompt: str, reference: str):
    generated = predict(prompt)
    scores = evaluate_all(reference, generated)
    return {"generated": generated, "scores": scores}
