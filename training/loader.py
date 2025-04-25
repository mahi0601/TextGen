import json
from torch.utils.data import DataLoader
from training.dataset import PromptDataset
from app.config import BATCH_SIZE


def load_training_data(data_path="data/train_data.txt"):
    with open(data_path, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

def get_dataloader():
    prompts = load_training_data()
    dataset = PromptDataset(prompts)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
