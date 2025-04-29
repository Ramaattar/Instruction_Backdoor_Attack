import fitz
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PdfDataset(Dataset):
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.df = pd.read_csv(csv_dir)
        self.pdf_paths = [row['out_path'] for _, row in self.df.iterrows()]
        # self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pdf_path = self.pdf_paths[idx]
        text = self.get_all_text(pdf_path)
        label = self.df.iloc[idx]['label']

        return text, label

    def get_all_text(self,input_pdf_path):
        """Get all text from a PDF using fitz (PyMuPDF)."""
        doc = fitz.open(input_pdf_path)
        res = ""
        for page in doc:
            text = page.get_text()
            res += text

        doc.close()
        return res

# Example usage
if __name__ == '__main__':
    data_dir = '/home/amohan2/cs680a/Instruction_Backdoor_Attack/data/output.csv'

    custom_dataset = PdfDataset(data_dir)
    dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

    # Iterate through the dataloader
    for text, labels in dataloader:
        print(text, labels)
        break
