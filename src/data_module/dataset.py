import os
import numpy as np
from pathlib import Path
import torch 
from dataclasses import dataclass
import torchaudio
import json
from torch.utils.data import Dataset

#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def read_jsonl(
    jsonl_file: str | Path, value_col: str
) :
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    """
    list = []
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            value = data[value_col]
            list.append(value)
    return list

class Pico_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 jsonl_file: str,
                 audio_column: str = "location",
                 text_column: str = "captions",
                 onset_column: str = "onset",
                 replace_column: str = "replace",
                 length_column: str = "length",
                 target_sr: int | None = None):
        
        super().__init__() 
        
        # Initialize class attributes
        self.jsonl_file = jsonl_file
        self.audio_column = audio_column
        self.text_column = text_column
        self.onset_column = onset_column
        self.replace_column = replace_column
        self.length_column = length_column
        self.target_sr = target_sr
        
        # Load data from jsonl file
        self.captions = read_jsonl(self.jsonl_file, self.text_column)
        self.audios = read_jsonl(self.jsonl_file, self.audio_column)
        self.onsets = read_jsonl(self.jsonl_file, self.onset_column)
        self.replace_labels = read_jsonl(self.jsonl_file, self.replace_column)
        self.lengths = read_jsonl(self.jsonl_file, self.length_column)
        self.indices = list(range(len(self.captions)))
        self.path=Path.cwd().parent

    def __len__(self):
        return len(self.captions)
    
    @property
    def task(self):
        return "picoaudio"

    def __getitem__(self, index):

        audio_path = self.path / self.audios[index] 
        #print(audio_path)
        audio_id = Path(audio_path).stem
        content  = {
            "caption": self.captions[index],
            "onset": self.onsets[index],
            "replace_label": self.replace_labels[index],
            "length": self.lengths[index]
        }

        condition = None
        waveform, orig_sr = torchaudio.load(audio_path)
        waveform = waveform.mean(0)

        if self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.target_sr)
        
        return {
            "audio_id": audio_id,
            "content": content,
            "waveform": waveform,
            "condition": condition,
            "task": self.task
        }

class AudioGenConcatDataset(Dataset):
    """
    Concatenates multiple Pico_Dataset datasets for joint access.

    Args:
        datasets (list[Pico_Dataset]): List of Pico_Dataset instances.
    """
    def __init__(self, datasets: list[Pico_Dataset]):
        self.datasets = datasets
        self.lengths = np.array([len(d) for d in datasets])
        self.cum_sum_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cum_sum_lengths - 1, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cum_sum_lengths[dataset_idx - 1]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]
    

if __name__ == "__main__":
    from tqdm import tqdm

    dataset = Pico_Dataset(
        jsonl_file = "/mnt/petrelfs/zhengzihao/workspace/pico/data/json/example.jsonl",
    )
    sample = dataset[0]
    print("=== Sample Structure ===")
    print(f"Returned data type: {type(sample)}")
    print(f"Keys: {sample.keys()}")

    print("\n=== Sample Details ===")
    for key, value in sample.items():
        if key == "waveform":
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, mean={value.mean():.3f}")
        elif key == "content":
            print(f"{key}:")
            for sub_key, sub_val in value.items():
                print(f"  {sub_key}: {sub_val[:50]}...") 
        else:
            print(f"{key}: {value}")