import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from app.text_generator import TextGenerator
from app.config import *
from training.dataset import PromptDataset

# Fix path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load training data (prompt||target)
with open("data/train_data.txt", "r", encoding="utf-8") as f:
    examples = [line.strip() for line in f.readlines() if "||" in line]

# Dataset and DataLoader
dataset = PromptDataset(examples)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextGenerator(pre_trained_model=MODEL_NAME).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

# Train
print("\n🚀 Starting training...")
model.train()
EPOCHS = 10  # Increased training epochs
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

# Save
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
print("Training complete!")
print("🚀 Model training complete!")