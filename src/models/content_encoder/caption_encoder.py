from typing import Any
import torch
import torch.nn as nn
import random
from utils.audiotime_event_merge import replace_event_synonyms

def decode_data(line_onset_str, latent_length):
    """
    Extracts a timestamp matrix (event onset indices) from a formatted onset string.

    Args:
        line_onset_str (str): String containing event names and onset intervals,
            formatted like "event1__start1-end1_start2-end2--event2__start1-end1".
        latent_length (int): Length of the output matrix.

    Returns:
        line_onset_index (torch.Tensor): Matrix of shape [4, latent_length], 
        line_event (list): List of event names extracted from the onset string.

    Notes:
        - 24000 is the audio sample rate.
        - 480 is the downsample ratio to align with VAE.
        - Each onset interval "start-end" (in seconds) is converted to embedding indices via (time * 24000 / 480).
    """
    line_onset_index = torch.zeros((4, latent_length)) # max for 4 events
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
    return line_onset_index, line_event
    

class ContentEncoder(nn.Module):
    """
    ContentEncoder encodes TCC and TDC information.
    """
    def __init__(
        self,
        text_encoder: nn.Module= None,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.pool = nn.AdaptiveAvgPool1d(1)

    def encode_content(
        self, batch_content: list[Any], batch_task: list[str],
        device: str | torch.device
    ):
        batch_output = []
        batch_mask = []
        batch_onset = []
        length_list = []

        for content, task in zip(batch_content, batch_task):
            if task == "picoaudio":

                caption = content["caption"]
                onset = content["onset"]
                length = int(float(content["length"]) *24000/480)
                # Replacement for AudioTime
                replace_label = content.get("replace_label", "False")
                if replace_label == "True":
                    caption, onset = replace_event_synonyms(caption, onset)
                
                # Handle random onset case for read data without timestamp
                if content["onset"] == "random":
                    length_list.append(length)
                    """
                    fixed embedding. Actually it's a sick sentence, a error during training, kept to match the checkpoint.
                    You can change it to sentence that difference to captions in datasets. 
                    The use of fixed text to obtain encoding is for numerical stability. 
                    We attempted to use learnable unified encoding during training, but the results were not satisfactory.
                    """
                    event = "There is no event here" 
                    event_embed = self.text_encoder([event.replace("_", " ")])["output"]
                    event_embed = self.pool(event_embed.permute(0, 2, 1))  # (B, 1024, 1)
                    event_embed = event_embed.flatten().unsqueeze(0)
                    new_onset = event_embed.repeat(length, 1).T
                else:
                    onset_matrix, events = decode_data(onset, length)
                    length_list.append(length)
                    new_onset = torch.zeros((1024, length), device=device) # 1024 for T5
                    # TDC
                    for (idx, event) in enumerate(events):
                        with torch.no_grad():
                            event_embed = self.text_encoder([event.replace("_", " ")])["output"]
                        event_embed = self.pool(event_embed.permute(0, 2, 1))  # (B, 1024, 1)
                        event_embed = event_embed.flatten().unsqueeze(0)
                        mask = (onset_matrix[idx, :] == 0)
                        cols = mask.nonzero(as_tuple=True)[0]
                        new_onset[:, cols] += event_embed.T.float()
                # TCC
                output_dict = self.text_encoder([caption])
            batch_output.append(output_dict["output"][0])
            batch_mask.append(output_dict["mask"][0])
            batch_onset.append(new_onset)
            
        # Pad all sequences in the batch to the same length for batching
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
