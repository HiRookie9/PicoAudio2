from typing import Any
import torch
import torch.nn as nn
import random
from utils.audiotime_event_merge import replace_event_synonyms
def decode_data(line_onset_str,train_loop,latent_length):
        if train_loop:
            line_onset_index = torch.zeros((4, latent_length))
            line_event = []
            event_idx = 0
            for event_onset in line_onset_str.split('--'):
                #print(event_onset)
                (event, instance) = event_onset.split('__')
                #print(instance)
                line_event.append(event)
                for start_end in instance.split('_'):
                    (start, end) = start_end.split('-')         
                    start, end = int(float(start)*24000/480), int(float(end)*24000/480)
                    if end > (latent_length - 1): break
                    line_onset_index[event_idx, start: end] = 1
                event_idx = event_idx + 1
            return line_onset_index, line_event, latent_length
        else:
            line_event = []
            length = int(10*24000/480)
            line_onset_index = torch.zeros((8, length))
            event_idx = 0
            for event_onset in line_onset_str.split('--'):
                (event, instance) = event_onset.split('__')
                line_event.append(event)
                for start_end in instance.split('_'):
                    (start, end) = start_end.split('-')         
                    start, end = int(float(start)*24000/480), int(float(end)*24000/480)
                    if end > (length - 1): break
                    line_onset_index[event_idx, start: end] = 1
                event_idx = event_idx + 1
            return line_onset_index, line_event, length
    

class ContentEncoder(nn.Module):
    def __init__(
        self,
        text_encoder: nn.Module= None,
        midi_encoder: nn.Module = None,
        pitch_encoder: nn.Module = None,
        audio_encoder: nn.Module = None
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.midi_encoder = midi_encoder
        self.pitch_encoder = pitch_encoder
        self.audio_encoder = audio_encoder
        self.pool = nn.AdaptiveAvgPool1d(1)  # 对第二维度池化
        #self.fc = nn.Linear(1024, 256)
        # 新增 embedding 层，大小为 (750, 1024)
        #self.onset_embedding = nn.Embedding(1, 1024)
    def encode_content(
        self, batch_content: list[Any], batch_task: list[str], train_loop, latent_length,
        device: str | torch.device
    ):
        batch_output = []
        batch_mask = []
        batch_onset = []
        length_list = []
        #print(batch_content)
        for content, task in zip(batch_content, batch_task):
            
            if task == "audio_super_resolution" or task =="speech_enhancement":
                content_dict = content
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                latent, latent_mask = self.audio_encoder.encode(content_dict["content"],content_dict["content_length"])
                output_dict = {"output": latent.transpose(1, 2), "mask": latent_mask}
            if task == "text_to_audio":
                output_dict = self.text_encoder([content])
            elif task == "picoaudio":
                caption = content["caption"]
                onset = content["onset"]
                replace_label = content.get("replace_label", "False")
                if replace_label == "True":
                    caption, onset = replace_event_synonyms(caption, onset)
                length = latent_length if isinstance(latent_length, int) else 500
                # 简化逻辑：onset != "random" 且 90% 概率走原方法，否则走embedding采样
                
                if onset == "random":
                    # 广播 embedding
                    #print("random")
                    #print(replace_label)
                    #print(onset)
                    length_list.append(length)
                    #new_onset = torch.zeros((1024, length), device=device)
                    event = "There is no event here"
                    event_embed = self.text_encoder([event.replace("_", " ")])["output"]
                    event_embed = self.pool(event_embed.permute(0, 2, 1))  # (B, 1024, 1)
                    event_embed = event_embed.flatten().unsqueeze(0)
                    #new_onset = self.onset_embedding(torch.zeros(1, dtype=torch.long, device=device))  # (1, 1024)
                    new_onset = event_embed.repeat(length, 1).T
                    #print("random")
                else:
                    onset_matrix, events, length = decode_data(onset, train_loop, latent_length)
                    #print("non_random")
                    #print(replace_label)
                    #print(onset)
                    length_list.append(length)
                    new_onset = torch.zeros((1024, length), device=device)
                    for (idx, event) in enumerate(events):
                        with torch.no_grad():
                            event_embed = self.text_encoder([event.replace("_", " ")])["output"]
                        event_embed = self.pool(event_embed.permute(0, 2, 1))  # (B, 1024, 1)
                        event_embed = event_embed.flatten().unsqueeze(0)
                        mask = (onset_matrix[idx, :] == 0)
                        cols = mask.nonzero(as_tuple=True)[0]
                        new_onset[:, cols] += event_embed.T.float()
                """
                length = int(10*24000/480)
                length_list.append(length)
                #new_onset = torch.zeros((1024, length), device=device)
                event = "There is no event here"
                event_embed = self.text_encoder([event.replace("_", " ")])["output"]
                event_embed = self.pool(event_embed.permute(0, 2, 1))  # (B, 1024, 1)
                event_embed = event_embed.flatten().unsqueeze(0)
                #new_onset = self.onset_embedding(torch.zeros(1, dtype=torch.long, device=device))  # (1, 1024)
                new_onset = event_embed.repeat(length, 1).T"""
                output_dict = self.text_encoder([caption])
            elif task == "singing_voice_synthesis":
                content_dict = {
                    "phoneme":
                        torch.as_tensor(content["phoneme"]).long(),
                    "midi":
                        torch.as_tensor(content["midi"]).long(),
                    "midi_duration":
                        torch.as_tensor(content["midi_duration"]).float(),
                    "is_slur":
                        torch.as_tensor(content["is_slur"]).long()
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                output_dict = self.midi_encoder(**content_dict)
            elif task == "singing_acoustic_modeling":
                content_dict = {
                    "phoneme": torch.as_tensor(content["phoneme"]).long(),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                content_dict["lengths"] = torch.as_tensor([
                    len(content["phoneme"])
                ])
                output_dict = self.pitch_encoder(**content_dict)

            batch_output.append(output_dict["output"][0])
            batch_mask.append(output_dict["mask"][0])
            batch_onset.append(new_onset)

        batch_output = nn.utils.rnn.pad_sequence(
            batch_output, batch_first=True, padding_value=0
        )
        batch_mask = nn.utils.rnn.pad_sequence(
            batch_mask, batch_first=True, padding_value=False
        )
        batch_onset = nn.utils.rnn.pad_sequence(
            batch_onset, batch_first=True, padding_value=0
        )
        return batch_output, batch_mask, batch_onset, length_list

    def encode_time_aligned_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_output = []

        for content, task in zip(batch_content, batch_task):
            
            if task =="audio_super_resolution":
                non_zero_positions = torch.nonzero(content, as_tuple=True)[0]
                content_length = non_zero_positions[-1] + 1
                latent, latent_mask = self.audio_encoder.encode(content.unsqueeze(0), content_length.unsqueeze(0))
                output_dict = {"output": latent.transpose(1, 2), "mask": latent_mask}
            if task == "text_to_audio":
                output_dict = {"output": torch.zeros(1, 1, device=device)}
            elif task == "speech_enhancement":
                output_dict = {"output": torch.zeros(1, 1, device=device)}
            elif task == "singing_voice_synthesis":
                output_dict = {"output": torch.zeros(1, 1, device=device)}
            elif task == "singing_acoustic_modeling":
                content_dict = {
                    "f0": torch.as_tensor(content["f0"]),
                    "uv": torch.as_tensor(content["uv"]),
                }
                for key in list(content_dict.keys()):
                    content_dict[key] = content_dict[key].unsqueeze(0).to(
                        device
                    )
                output_dict = self.pitch_encoder.encode_pitch(**content_dict)

            batch_output.append(output_dict["output"][0])

        batch_output = nn.utils.rnn.pad_sequence(
            batch_output, batch_first=True, padding_value=0
        )
        return batch_output
