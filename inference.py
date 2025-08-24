from pathlib import Path
import os
import soundfile as sf
import torch
import hydra
from omegaconf import OmegaConf
from safetensors.torch import load_file
import diffusers.schedulers as noise_schedulers
from tqdm import tqdm
import numpy as np
from utils.config import register_omegaconf_resolvers
from models.common import LoadPretrainedBase

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

register_omegaconf_resolvers()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = []

    @hydra.main(config_path="configs", config_name="inference")
    def parse_config_from_command_line(config):
        config = OmegaConf.to_container(config, resolve=True)
        configs.append(config)

    parse_config_from_command_line()
    config = configs[0]

    if "exp_dir" in config:
        exp_dir = Path(config["exp_dir"])
        ckpt_path: Path = sorted((exp_dir / "checkpoints").iterdir()
                                )[-1] / "model.safetensors"
    elif "ckpt_dir" in config:
        ckpt_dir = Path(config["ckpt_dir"])
        ckpt_path = ckpt_dir / "model.safetensors"
        exp_dir = ckpt_dir.parent.parent

    exp_config = OmegaConf.load(exp_dir / "config.yaml")
    model: LoadPretrainedBase = hydra.utils.instantiate(exp_config["model"])
    state_dict = load_file(ckpt_path)
    model.load_pretrained(state_dict)

    model = model.to(device)
    test_dataloader = hydra.utils.instantiate(
        config["test_dataloader"], _convert_="all"
    )
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

    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            for key in list(batch.keys()):
                data = batch[key]
                if isinstance(data, torch.Tensor):
                    batch[key] = data.to(device)

            waveform = model.inference(
                scheduler=scheduler,
                # latent_shape=config["latent_shape"],
                num_steps=config["num_steps"],
                guidance_scale=config["guidance_scale"],
                use_gt_duration=config["use_gt_duration"],
                **batch
            )

            if isinstance(batch["content"][0], str):
                out_file: str = batch["content"][0]
                #out_file: str = batch["audio_id"][0]
            else:
                out_file: str = batch["content"][0]["caption"]
                #out_file: str = batch["audio_id"][0]
                #print(out_file)
            if not out_file.endswith(".wav"):
                out_file = f"{out_file}.wav"
            
            out_file = out_file.replace('/', '_')
            # 检查重名，并加编号
            base_name, ext = os.path.splitext(out_file)
            candidate = out_file
            num = 1
            while (audio_output_dir / candidate).exists():
                candidate = f"{base_name}---{num}{ext}"
                num += 1
            out_file = candidate
            """
            audio = waveform[0, 0].cpu().numpy()
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            audio = np.clip(audio, -1, 1)
            sf.write(
                audio_output_dir / out_file,
                audio,
                samplerate=exp_config["sample_rate"],
            )"""
            sf.write(
                audio_output_dir / out_file,
                waveform[0, 0].cpu().numpy(),
                samplerate=exp_config["sample_rate"],
            )


if __name__ == "__main__":
    main()
