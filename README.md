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

- FLAN-T5 path [https://huggingface.co/google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
- StableVAE path
- Noise scheduler path
- PicoAudio2 experiment/checkpoint path (to be released)
- Dataset path

### 2. Run Batch Inference

After configuration, run batch inference with:

```bash
cd src
bash bash_scripts/test.sh
```

### 3. Single Data Inference or LLM-based TDC Generation

- Enter your LLM API key in `utils/llm.py`.
- Edit `configs/inference_llm.yaml` to set model and data paths.

To run LLM-based inference, use:

```bash
cd src
bash bash_scripts/test_llm.sh
```

## Training

#### Autoencoder
Refer to the VAE training section in our work [SoloAudio](https://github.com/WangHelin1997/SoloAudio)

#### T2A Diffusion Model
Prepare your data and pretrained models, then run:

```bash
cd src
bash bash_scripts/train_pico_4gpus.sh
```

## Todo
- [ ]  Release training dataset and checkpoints
- [ ]  Release Gradio Demo along with checkpoints

## Author's Note

Thank you for your attention and use!  
This is my first open-source project. The code has been refined for simplicity and readability before release.  
If you encounter any problems or have questions, please open an issue on GitHub, or contact me via email(rookie9@sjtu.edu.cn). I will respond as soon as possible.

---


