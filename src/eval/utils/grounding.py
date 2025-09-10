import re
import os
import json
import torch
import torchaudio
from tqdm import tqdm

#os.environ['TRANSFORMERS_OFFLINE'] = '0'
#os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModel

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    "wsntxxn/cnn8rnn-w2vmean-audiocaps-grounding",
    trust_remote_code=True
).to(device)


def merge_adjacent_segments(event_times, max_gap=1.0):
    """
    Merge adjacent time segments if the gap between them is less than max_gap.

    Args:
        event_times (list[str]): List of time ranges, e.g. ["0.00-1.00", "1.10-2.00"]
        max_gap (float): Maximum allowed gap between segments to merge.

    Returns:
        list[str]: List of merged time ranges.
    """
    merged_times = []
    start_time, end_time = None, None

    for time_range in event_times:
        start, end = map(float, time_range.split('-'))
        if start_time is None:
            # Initialize the first segment
            start_time, end_time = start, end
        elif start - end_time <= max_gap:
            end_time = end
        else:
            # Push the previous segment and start a new one
            merged_times.append(f"{start_time:.3f}-{end_time:.3f}")
            start_time, end_time = start, end

    # Append the last segment
    if start_time is not None:
        merged_times.append(f"{start_time:.3f}-{end_time:.3f}")

    return merged_times

def extract_onset_times(output, threshold=0.5, max_gap=1.0):
    """
    Extract onset time segments from model output based on a threshold,
    and merge adjacent segments.

    Args:
        output (torch.Tensor): Model output, shape [num_events, time_steps].
        threshold (float): Activation threshold for event.
        max_gap (float): Maximum gap to merge adjacent segments.

    Returns:
        list[str]: Onset time segments for each event.
    """
    onset_times = []
    num_events, time_steps = output.shape
    for event_idx in range(num_events):
        event_times = []
        start_time = None
        for time_idx in range(time_steps):
            if output[event_idx, time_idx] >= threshold:
                if start_time is None:
                    start_time = time_idx / 25.0  # Convert frame to seconds
            else:
                if start_time is not None:
                    end_time = time_idx / 25.0
                    event_times.append(f"{start_time:.3f}-{end_time:.3f}")
                    start_time = None
        if start_time is not None:  # Handle ongoing event
            event_times.append(f"{start_time:.3f}-{time_steps / 25.0:.3f}")
        merged_times = merge_adjacent_segments(event_times, max_gap)
        onset_times.append("_".join(merged_times))
    return onset_times

def process_audio_with_model(audio_path, text):
    """
    Load audio and process it with the model.

    Args:
        audio_path (str or Path): Path to audio file.
        text (list[str]): List of event names in natural language.

    Returns:
        torch.Tensor: Model output, shape [num_events, time_steps].
    """
    wav, sr = torchaudio.load(audio_path)
    wav = torchaudio.functional.resample(wav, sr, model.config.sample_rate)
    wav = wav.mean(0) if wav.size(0) > 1 else wav[0]
    with torch.no_grad():
        output = model(
            audio=wav.unsqueeze(0),
            audio_len=wav.size(0),
            text=text
        )
    # output: shape [num_events, time_steps]
    return output

def groundtruth_inference(audio_path, event_list, threshold=0.5, max_gap=1.0):
    """
    Perform groundtruth inference for audio and event list.

    Args:
        audio_path (str): Path to audio file.
        event_list (list[str]): List of event names with underscores.
        threshold (float): Activation threshold.
        max_gap (float): Maximum gap to merge segments.

    Returns:
        str: Onset-style string result, e.g. "event__0.00-1.00--event2__1.10-2.00"
    """
    # Convert event names to natural language for model input
    text_input = [e.replace("_", " ") for e in event_list]
    output = process_audio_with_model(audio_path, text_input)
    
    if hasattr(output, "detach"):  # 如果 output 是Tensor
        output = output.detach().cpu()
    elif hasattr(output, "output"):
        output = output.output.detach().cpu()
    else:
        output = torch.tensor(output)
    onset_segs = extract_onset_times(output, threshold=threshold, max_gap=max_gap)
    
    # Compose output string with original event names and onset segments
    onset_str = "--".join(
        f"{event}__{seg}" for event, seg in zip(event_list, onset_segs) if seg and seg != ""
    )
    return onset_str

def main(gen_jsonl_path, caption_jsonl_path, onset_jsonl_path, output_jsonl_path, threshold=0.5):
    """
    Run groundtruth inference on a batch of audio files.
    """

    with open(gen_jsonl_path, 'r') as f:
        gen_data = {item["audio_id"]: item["audio"] for item in map(json.loads, f)}
    with open(caption_jsonl_path, 'r') as f:
        captions = {item["audio_id"]: item["caption"] for item in map(json.loads, f)}
    with open(onset_jsonl_path, 'r') as f:
        onsets = {item["audio_id"]: item["onset"] for item in map(json.loads, f)}

    output_data = []
    for audio_id in tqdm(gen_data, desc="Groundtruth inference"):
        gen_audio_path = gen_data[audio_id]
        caption = captions.get(audio_id, "")
        onset_str = onsets.get(audio_id, "")
        if not onset_str or not caption:
            continue
        # Parse groundtruth event list from onset string
        gt_event_list = []
        for event_chunk in onset_str.strip().split("--"):
            if not event_chunk:
                continue
            event_name = event_chunk.split("__")[0]
            gt_event_list.append(event_name)
        if not gt_event_list:
            continue
        # infer
        pred_onset_str = groundtruth_inference(gen_audio_path, gt_event_list, threshold=threshold)
        output_data.append({
            "audio_id": audio_id,
            "audio": gen_audio_path,
            "caption": caption,
            "pred_onset": pred_onset_str
        })

    with open(output_jsonl_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # 路径根据你的实际情况修改
    gen_jsonl_path = ""
    caption_jsonl_path = ""
    onset_jsonl_path = ""
    output_jsonl_path = ""
    main(gen_jsonl_path, caption_jsonl_path, onset_jsonl_path, output_jsonl_path, threshold=0.5)