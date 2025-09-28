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

### 1. Configure Model and Dataset Paths

Before running inference, **edit your configuration files to set the correct local paths** for required model checkpoints and data:

- FLAN-T5 path
- StableVAE path
- Noise scheduler path
- PicoAudio2 experiment/checkpoint path (to be released)
- Dataset path

### 2. Run Batch Inference

After configuration, run batch inference with:

```bash
bash bash_scripts/test.sh
```

### 3. Single Data Inference or LLM-based TDC Generation

- Enter your LLM API key in `utils/llm.py`.
- Edit `configs/inference_llm.yaml` to set model and data paths.

To run LLM-based inference, use:

```bash
bash bash_scripts/test_llm.sh
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
