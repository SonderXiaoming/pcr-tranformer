import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np


def get_min_dict(path):
    temp = {}
    with open(path, "r") as f:
        temp = json.load(f)
    return temp


min_dict: dict = get_min_dict(Path(__file__).parent / "min_unit.json")
reverse_min_dict = {v: int(k) for k, v in min_dict.items()}
len_min = len(min_dict)


class LabelTransform:
    def __init__(self, size: int, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label: np.ndarray):
        label = np.pad(
            label,
            (0, (self.size - label.shape[0])),
            mode="constant",
            constant_values=self.pad,
        )
        return label


class PCRJJCDataset(Dataset):
    def __init__(self, path: Path):
        self.transform = LabelTransform(55, reverse_min_dict[9999])
        with open(path, "r") as f:
            self.data: dict = json.load(f)
            self.keys = list(self.data.keys())
        # self.unit_dict = self.get_dict()
        self.dict_len = len_min

    def __len__(self):
        # Returns the size of the dataset
        return len(self.keys)

    def __getitem__(self, idx):
        # Retrieve the key (input) at the specified index
        key = self.keys[idx]
        input_sequence = (
            [reverse_min_dict[250]]
            + [reverse_min_dict[int(k)] for k in key.split(",")]
            + [reverse_min_dict[520]]
        )
        input_sequence = self.transform(np.asarray(input_sequence))

        output_sequences = [reverse_min_dict[250]]
        for atk_team in self.data[key]:
            output_sequences += [reverse_min_dict[atk] for atk in atk_team]
        output_sequences.append(reverse_min_dict[520])
        output_sequences = self.transform(np.asarray(output_sequences))
        return torch.LongTensor(input_sequence), torch.LongTensor(output_sequences)
