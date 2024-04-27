import pandas as pd
import transformers
import torch

class PairsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer: transformers.AutoTokenizer,
                 path: str=None,
                 data: pd.DataFrame=None,
                 filter_same: bool=True,
                 SEQ_LEN: int=64):
        if data is None:
            data = pd.read_csv(path, index_col=0)
        self.dataset = data[(data.was_changed) | (not filter_same)].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.SEQ_LEN = SEQ_LEN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: list[int]):
        text1 = self.tokenizer(self.dataset.loc[idx]['base'],
                     padding='max_length',
                     truncation=True,
                     max_length=self.SEQ_LEN,)

        text2 = self.tokenizer(self.dataset.loc[idx]['polypers'],
                     padding='max_length',
                     truncation=True,
                     max_length=self.SEQ_LEN,)

        if "base_mask" in self.dataset.columns:
            return text1, text2, list(map(int, self.dataset.loc[idx]["base_mask"].strip("][").split(" "))), list(map(int, self.dataset.loc[idx]["polypers_mask"].strip("][").split(" ")))

        return text1, text2

