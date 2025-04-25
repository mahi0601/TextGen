from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from app.config import MODEL_NAME, MAX_LENGTH

class PromptDataset(Dataset):
    def __init__(self, examples):
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.examples = examples
        self.max_length = MAX_LENGTH

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        prompt, target = item.split("||")  # prompt||target in train_data.txt
        full_input = prompt.strip() + " " + target.strip()

        tokens = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': tokens['input_ids'].squeeze(0)
        }
