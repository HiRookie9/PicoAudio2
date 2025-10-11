# PicoAudio2: Temporal Controllable Text-to-Audio Generation with Natural Language Description
[![Official Page](https://img.shields.io/badge/Official%20Page-PicoAudio2-blue?logo=Github&style=flat-square)](https://hirookie9.github.io/PicoAudio2-Page/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.00683-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.00683)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/rookie9/PicoAudio2)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wsntxxn/PicoAudio2)  

🟣 PicoAudio2 is a temporal controllable Text-to-Audio model with natural language description.


## Quick start
You can see the demo on the website [Huggingface Online Inference](https://huggingface.co/spaces/wsntxxn/PicoAudio2) and [Github Demo](https://hirookie9.github.io/PicoAudio2-Page/).

Alternatively, you can generate samples as follows:
```bash
# Install other dependencies
git clone -b infer https://github.com/HiRookie9/PicoAudio2.git
pip install -r requirements.txt
```

You can quickly generate audio with the following code:
```python
import torch
import soundfile as sf
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("rookie9/PicoAudio2", trust_remote_code=True).to(device)

content = {
    "caption": "a dog barks",
    "onset": "a_dog_barks__1.0-2.0_3.0-4.0",
    "length": 5.0
}

with torch.no_grad():
    waveform = model(content)
    sf.write("output.wav", waveform[0, 0].cpu().numpy(), samplerate=24000)
```

Alternatively, you can use the script *"utils/infer.py"* to infer with llm (please enter your API key in *"utils/llm.py"*).

There are still some bugs when loading checkpoints with AutoModel, which may slightly reduce sound quality. If you are pursuing a better user experience or evaluating models, please use the following code instead. We will fix this issue before 2025/10/18.

---

## Installation

Clone the repository:
```
git clone https://github.com/HiRookie9/PicoAudio2.git
```

### 1. Create and Activate Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n picoaudio2 python=3.10 -y

# Activate the environment
conda activate picoaudio2
```

### 2. Install PyTorch and Related Packages

```bash
# Install PyTorch, TorchAudio, TorchVision, and TorchData (CUDA 11.8)
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 torchdata==0.11.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Install Other Requirements

```bash
# Install other dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Configure Model and Dataset Paths

Before running inference, **edit your configuration files to set the correct local paths** for required model checkpoints and data:

- FLAN-T5: [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)
- VAE: [OpenSound/EzAudio](https://huggingface.co/OpenSound/EzAudio/tree/main/ckpts/vae)
- Noise scheduler: [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1/tree/main/scheduler)
- PicoAudio2 experiment/checkpoint path (to be released)
- Test Dataset path

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
Prepare your data and pretrained models, then run:

```bash
cd src
bash bash_scripts/train_pico_4gpus.sh
```

## Todo
- [ ]  Release post-processing code for eval
- [ ]  Release full dataset and checkpoints
- [ ]  Release Gradio Demo along with checkpoints(fixing the bug in the checkpoint, which leads lower quality)

## Author's Note

Thank you for your attention and use!  
This is my first open-source project. The code has been refined for simplicity and readability before release.  
If you encounter any problems or have questions, please open an issue on GitHub, or contact me via email(rookie9@sjtu.edu.cn). I will respond as soon as possible.

## Acknowledgement
Thanks for these works: [UniFlow-Audio](https://github.com/wsntxxn/UniFlow-Audio), [PicoAudio](https://github.com/zeyuxie29/PicoAudio), [EzAudio](https://github.com/haidog-yaqub/EzAudio), [audioldm_eval](https://github.com/haoheliu/audioldm_eval), [TAG](https://github.com/wsntxxn/TextToAudioGrounding)
