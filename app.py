import streamlit as st
import torch
from model.text_generator import TextGenerator
from model.utils import tokenize_text, evaluate_output

# Model and tokenizer initialization
model = TextGenerator(hidden_dim=256, num_layers=2)
tokenizer = model.tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Streamlit app interface
st.title("TEXT-GENERATOR")
st.write("Generate text with custom prompts!")

prompt = st.text_input("Enter your prompt:")
if st.button("Generate Text"):
    input_ids = tokenize_text(prompt, tokenizer).to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    
    generated_tokens = torch.argmax(outputs, dim=-1)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    st.write("Generated Text:")
    st.write(generated_text)
    
    # Evaluate output if reference text is provided
    reference = st.text_input("Enter reference text (for evaluation):")
    if reference:
        evaluation = evaluate_output(reference, generated_text)
        st.write("Evaluation Metrics:")
        st.write(f"BLEU Score: {evaluation['BLEU']}")
        st.write(f"ROUGE Scores: {evaluation['ROUGE']}")
