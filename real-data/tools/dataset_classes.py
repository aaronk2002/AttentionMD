from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("../../tokenizer")


class IMDb(Dataset):
    def __init__(self, train, max_length=512, device="cpu"):
        set_dir = "train" if train else "test"
        pos_files = os.listdir(f"../../dataset/imdb_larger/{set_dir}/pos")
        neg_files = os.listdir(f"../../dataset/imdb_larger/{set_dir}/neg")
        self.review = []
        self.label = []

        # Get the positive files
        for file in pos_files:
            f = open(
                f"../../dataset/imdb_larger/{set_dir}/pos/{file}", "r", encoding="UTF-8"
            )
            self.review.append(
                tokenizer(f.read(), padding="max_length", max_length=max_length)[
                    "input_ids"
                ][:max_length]
            )
            self.label.append(True)

        # Get the negative files
        for file in neg_files:
            f = open(
                f"../../dataset/imdb_larger/{set_dir}/neg/{file}", "r", encoding="UTF-8"
            )
            self.review.append(
                tokenizer(f.read(), padding="max_length", max_length=max_length)[
                    "input_ids"
                ][:max_length]
            )
            self.label.append(False)

        # Tensorize
        self.review = torch.tensor(self.review, dtype=torch.int64).to(device)
        self.label = torch.tensor(self.label).to(device)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.review[index], self.label[index]
