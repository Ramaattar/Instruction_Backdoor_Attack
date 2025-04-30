import fitz
import pandas as pd
from utils.instructions import instructions
from torch.utils.data import Dataset
import torch

class PdfDataset(Dataset):
    def __init__(self, csv_dir, tokenizer):
        self.df = pd.read_csv(csv_dir)
        self.pdf_paths = self.df['out_path'].tolist()
        self.labels = self.df['label'].tolist()
        self.instructions_ = instructions(
            dataset="pdf_dataset",
            attack_type='word',
            trigger_word="cf",
            target_label=0
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pdf_path = self.pdf_paths[idx]
        label = self.labels[idx]
        text = self.get_all_text(pdf_path)
        text = self.instructions_['instruction'] + text + self.instructions_['end']
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)  # remove batch dimension
        return input_ids, torch.tensor(label)

    def get_all_text(self, input_pdf_path):
        doc = fitz.open(input_pdf_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return text
