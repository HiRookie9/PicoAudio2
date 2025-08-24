import os
import numpy as np
from pathlib import Path
import torch 
from dataclasses import dataclass
#import pandas as pd
import torchaudio
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def read_jsonl(
    jsonl_file: str | Path, value_col: str
) :
    """
    Read two columns, indicated by `key_col` and `value_col`, from the
    given jsonl file to return the mapping dict
    TODO handle duplicate keys
    """
    list = []
    with open(jsonl_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line.strip())
            value = data[value_col]
            list.append(value)
    return list


class Text_Onset_2_Audio_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 jsonl_file: str,  # 必须显式声明参数
                 audio_column: str = "location",
                 text_column: str = "captions",
                 onset_column: str = "onset",
                 target_sr: int | None = None):
        
        super().__init__()  # 必须调用父类初始化
        
        # 初始化类属性
        self.jsonl_file = jsonl_file
        self.audio_column = audio_column
        self.text_column = text_column
        self.onset_column = onset_column
        self.target_sr = target_sr
        
        # 加载数据
        self.captions = read_jsonl(self.jsonl_file, self.text_column)
        self.audios = read_jsonl(self.jsonl_file, self.audio_column)
        self.onsets = read_jsonl(self.jsonl_file, self.onset_column)
        self.indices = list(range(len(self.captions)))

    def __len__(self):
        return len(self.captions)
    
    @property
    def task(self):
        return "picoaudio"

    def __getitem__(self, index):
        audio_path = self.audios[index]
        audio_id = Path(audio_path).stem
        content  = {
            "caption": self.captions[index],
            "onset": self.onsets[index]
        }
        condition = None
        waveform, orig_sr = torchaudio.load(self.audios[index])
            # average multi-channel to single-channel
        waveform = waveform.mean(0)

        if self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=orig_sr, new_freq=self.target_sr)
        duration =[1.0]
        return {
            "audio_id": audio_id,
            "content": content,
            "waveform": waveform,
            "condition": condition,
            "duration": duration,
            "task": self.task
        }
"""
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        batch = []
        for i in dat:     
            if i==1:
                batch.append(torch.tensor(np.array(dat[i].tolist()), dtype=torch.float32))
            elif i==2:
                batch.append(torch.tensor(dat[i]))
            else:
                batch.append(dat[i].tolist())
        return batch"""
    

if __name__ == "__main__":
    from tqdm import tqdm

    dataset = Text_Onset_2_Audio_Dataset(
    jsonl_file = "/hpc_stor03/sjtu_home/zihao.zheng/cag/data_utils/meta_data/train_multi-event_v3.json",
    )
    sample = dataset[0]
    print("=== 样本结构 ===")
    print(f"返回数据类型: {type(sample)}")
    print(f"包含键: {sample.keys()}")

    print("\n=== 详细内容 ===")
    for key, value in sample.items():
        if key == "waveform":
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, mean={value.mean():.3f}")
        elif key == "content":
            print(f"{key}:")
            for sub_key, sub_val in value.items():
                print(f"  {sub_key}: {sub_val[:50]}...")  # 截取前50字符防止过长
        else:
            print(f"{key}: {value}")
"""
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        sample = dataset[i]
    
        content_last = sample['content'][-1] if isinstance(sample['content'], (list, torch.Tensor)) else None
        waveform_last = sample['waveform'][-1] if isinstance(sample['waveform'], (list, torch.Tensor)) else None
        if content_last == 0 and waveform_last == 0:
            print(f"Sample {i}:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor shape {value.shape}")
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
"""