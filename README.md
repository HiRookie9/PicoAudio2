# PicoAudio2: Temporal Controllable Text-to-Audio Generation with Natural Language Description
[![Official Page](https://img.shields.io/badge/Official%20Page-PicoAudio2-blue?logo=Github&style=flat-square)](https://hirookie9.github.io/PicoAudio2-Page/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.00683-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.00683)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://hirookie9.github.io/PicoAudio2-Page/)

🟣 PicoAudio2 is a temporal controllable Text-to-Audio model with natural language description.


## Installation

Clone the repository:
```
git clone https://github.com/HiRookie9/PicoAudio2.git
```
Install the dependencies:
```
cd PicoAudio2/src
pip install -r requirements.txt
```


## Usage

You can use the model with the following code:

```python
from api.ezaudio import EzAudio
import torch
import soundfile as sf

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ezaudio = EzAudio(model_name='s3_xl', device=device)

# text to audio genertation
prompt = "a dog barking in the distance"
sr, audio = ezaudio.generate_audio(prompt)
sf.write(f'{prompt}.wav', audio, sr)

# audio inpainting
prompt = "A train passes by, blowing its horns"
original_audio = 'ref.wav'
sr, audio = ezaudio.editing_audio(prompt, boundary=2, gt_file=original_audio,
                                  mask_start=1, mask_length=5)
sf.write(f'{prompt}_edit.wav', audio, sr)
```

## Training

#### Autoencoder
Refer to the VAE training section in our work [SoloAudio](https://github.com/WangHelin1997/SoloAudio)

#### T2A Diffusion Model
Prepare your data (see example in `src/dataset/meta_example.csv`), then run:

```bash
cd src
accelerate launch train.py
```

## Todo
- [x] Release Gradio Demo along with checkpoints [EzAudio Space](https://huggingface.co/spaces/OpenSound/EzAudio)
- [x] Release ControlNet Demo along with checkpoints [EzAudio ControlNet Space](https://huggingface.co/spaces/OpenSound/EzAudio-ControlNet)
- [x] Release inference code
- [x] Release training pipeline and dataset
- [x] Improve API and support automatic ckpts downloading 
- [ ] Release checkpoints for stage1 and stage2 [WIP]

## Reference

If you find the code useful for your research, please consider citing:

```bibtex
@article{hai2024ezaudio,
  title={EzAudio: Enhancing Text-to-Audio Generation with Efficient Diffusion Transformer},
  author={Hai, Jiarui and Xu, Yong and Zhang, Hao and Li, Chenxing and Wang, Helin and Elhilali, Mounya and Yu, Dong},
  journal={arXiv preprint arXiv:2409.10819},
  year={2024}
}
```

## Acknowledgement
Some codes are borrowed from or inspired by: [U-Vit](https://github.com/baofff/U-ViT), [Pixel-Art](https://github.com/PixArt-alpha/PixArt-alpha), [Huyuan-DiT](https://github.com/Tencent/HunyuanDiT), and [Stable Audio](https://github.com/Stability-AI/stable-audio-tools).
