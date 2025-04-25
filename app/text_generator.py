import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TextGenerator(nn.Module):
    def __init__(self, pre_trained_model="gpt2"):
        super(TextGenerator, self).__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(pre_trained_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = GPT2LMHeadModel.from_pretrained(pre_trained_model)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def generate_text(self, prompt, max_length=100):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=1.0,
                early_stopping=True
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
