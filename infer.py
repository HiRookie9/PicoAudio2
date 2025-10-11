import torch
import json
import soundfile as sf
from transformers import AutoModel
from utils.llm import get_time_info

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("rookie9/PicoAudio2", trust_remote_code=True).to(device)

def a_to_b(a_str):
    items = a_str.split(';')
    result = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if '(' in item and ')' in item:
            name, times = item.split('(', 1)
            name = name.strip().replace(' ', '_')
            times = times.strip(')').replace(', ', '_').replace(',', '_')
            result.append(f"{name}__{times}")
    return '--'.join(result)

input_text = "a dog barks"
input_onset = "a dog barks(3.0-4.0, 6.0-7.0)"
input_length = "10.0"

# using llm
# input_json = json.loads(get_time_info(input_text))
# input_onset, input_length = input_json["onset"], input_json["length"]

content = {
        "caption": input_text,
        "onset": a_to_b(input_onset),
        "length": input_length
    }
    
with torch.no_grad():
    waveform = model(content)
    output_wav = "output.wav"
    sf.write(
        output_wav,
        waveform[0, 0].cpu().numpy(),
        samplerate=24000,
    )
