from pathlib import Path
import os
import json
import soundfile as sf
import torch
import hydra
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
import numpy as np
from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase

# Choose the appropriate way to call LLM API for you
from utils.llm import get_time_info


try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()

@hydra.main(config_path="configs", config_name="inference_llm")
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir.parent.parent
    elif "exp_dir" in config:
        exp_dir = Path(config["exp_dir"])
        ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir())[-1] / "model.safetensors"


    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)
    model = model.to(device)
    model.eval()

    scheduler = getattr(
        noise_schedulers,
        config["noise_scheduler"]["type"],
    ).from_pretrained(
        config["noise_scheduler"]["name"],
        subfolder="scheduler",
    )

    audio_output_dir = exp_dir / config["wav_dir"]
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    # construct data based on input
    input_text = config.get("input_text", "a dog barks")
    time_control = config.get("time_control", False)
    input_onset = config.get("input_onset", None)
    input_length = config.get("input_length", None)

    if time_control:
        if input_onset is None or input_length is None:
            input_json = json.loads(get_time_info(input_text))
            input_onset, input_length = input_json["onset"], input_json["length"]
    else:
        if input_onset is None:
            input_onset = "random"
        if input_length is None:
            input_length = "10.0"

    content = {
        "caption": input_text,
        "onset": input_onset,
        "length": input_length
    }
    batch = {
        "content": [content],
        "condition": None,
        "task": ["picoaudio"]
    }

    for key in list(batch.keys()):
        data = batch[key]
        if isinstance(data, torch.Tensor):
            batch[key] = data.to(device)

    with torch.no_grad():
        waveform = model.inference(
            scheduler=scheduler,
            num_steps=config["num_steps"],
            guidance_scale=config["guidance_scale"],
            **batch
        )

        out_file = input_text
        if not out_file.endswith(".wav"):
            out_file = f"{out_file}.wav"
        out_file = out_file.replace('/', '_')
        base_name, ext = os.path.splitext(out_file)
        candidate = out_file
        num = 1
        while (audio_output_dir / candidate).exists():
            candidate = f"{base_name}---{num}{ext}"
            num += 1
        out_file = candidate

        sf.write(
            audio_output_dir / out_file,
            waveform[0, 0].cpu().numpy(),
            samplerate=exp_config["sample_rate"],
        )

if __name__ == "__main__":
    main()