from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import torch

class ISWLTDataset(Dataset):
    def __init__(self, path, en_de = True, max_len =128):
        if en_de == True:

            path = os.path.join(path, 'iwslt2017/en-de')
            en_path = os.path.join(path, 'train.tags.en-de.en')
            de_path = os.path.join(path, 'train.tags.en-de.de')
            self.src_data = self.read_and_clean(en_path)
            self.tgt_data = self.read_and_clean(de_path)
            self.src_lan = 'en'
            self.tgt_lan = 'de'

        else:
            path = os.path.join(path, 'iwslt2017/de-en')
            de_path = os.path.join(path, 'train.tags.de-en.de')
            en_path = os.path.join(path, 'train.tags.de-en.en')
            self.src_data = self.read_and_clean(de_path)
            self.tgt_data = self.read_and_clean(en_path)
            self.src_lan = 'de'
            self.tgt_lan = 'en'

        assert len(self.src_data) == len(self.tgt_data), 'Source data and target data must have same length.'
        self.tokenizer = AutoTokenizer.from_pretrained('./t5')
        self.max_len = max_len

    def read_and_clean(self, path):
        lines = []
        with open(path, 'r', encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()  # 开头结尾空格符
                if line.startswith("<"):
                    continue # 跳过xml标签行
                if line:
                    lines.append(line)
        return lines


    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):

        src = self.src_data[idx]
        tgt = self.tgt_data[idx]

        src_enc = self.tokenizer(src, max_length=self.max_len, truncation=True, padding="max_length", return_tensors=None)
        tgt_enc = self.tokenizer(tgt, max_length=self.max_len, truncation=True, padding="max_length", return_tensors=None)

       
        return {
            "input_ids": torch.tensor(src_enc['input_ids']).squeeze(0),
            "src_attn_mask": torch.tensor(src_enc['attention_mask']).squeeze(0),
            'labels': torch.tensor(tgt_enc['input_ids']).squeeze(0),
            'tgt_attn_mask': torch.tensor(tgt_enc['attention_mask']).squeeze(0)
        }

def ISWLTDataLoader(cfg):
    iswlt = ISWLTDataset(cfg.DATASET.PATH, cfg.DATASET.EN_DE, cfg.DATASET.BPTT)
    dataloader = DataLoader(iswlt, batch_size = cfg.TRAIN.BATCH_SIZE, shuffle = True, num_workers = 4)
    return dataloader, len(iswlt.tokenizer)
