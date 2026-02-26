# 🚀 HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models

**CVPR 2026** [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Yangguang Lin, Quan Fang, Yufei Li, Jiachen Sun, Junyu Gao, Jitao Sang

---

## 🎯 Overview

We introduce a novel method named **HulluEdit** (Evidence-Consistent Subspace Editing), which can effectively mitigate object hallucinations (OH) in Large Vision-Language Models (LVLMs). HulluEdit edits model weights by extracting visual evidence subspaces and orthogonalizing the model behavior based on subspace projection:

- 🔬 **Evidence Subspace Extraction**: Constructs a visual evidence subspace from image-conditioned token representations by analyzing hidden states of truthful vs. hallucinated samples through SVD, identifying main directions of visual grounding
- 🛡️ **Anti-Prior Subspace Construction**: Builds an anti-prior subspace to suppress language-only biases that lead to hallucinations
- ✏️ **Closed-Form Weight Editing**: Applies closed-form edit to MLP weights in the LLM backbone, projecting to evidence subspace while suppressing anti-prior directions

![Model Diagram](docs/model.png)

---

## ✨ Key Features

- 🎨 **Single-Pass Editing** - Edit model weights in one forward pass
- 🔒 **Evidence-Based** - Grounded in visual evidence extraction
- 🚀 **Closed-Form Solution** - No iterative optimization required
- 📈 **Effective** - Significantly reduces object hallucinations

---

## 🛠️ Getting Started

### 📦 Environment Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/HulluEdit.git
cd HulluEdit

# Create and activate environment
conda create -n hullu python=3.10
conda activate hullu

# Install dependencies
pip install -r requirements.txt
```

### 🤖 Model Setup

Prepare the following model checkpoints:

| Model | Download Link |
|:---|:---|
| 🟦 **LLaVA-1.5 7B** | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| 🟦 **LLaVA-1.5 13B** | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| 🟨 **MiniGPT-4 (LLaMA-2 Chat 7B)** | [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) |
| 🟨 **MiniGPT-4 LLM** | [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| 🟪 **mPLUG-Owl2** | [MAGAer13/mplug-owl2-llama2-7b](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b) |
| 🟧 **QwenVL** | [Qwen/Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) |
| 🟧 **QwenVL 2.5** | [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) |

### 📊 Dataset Preparation

#### MSCOCO 2014

Download from [COCO Dataset](https://cocodataset.org/#download) and organize as:

```
DATA/
├── annotations/
│   ├── instances_val2014.json
│   └── captions_val2014.json
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    └── ...
```

#### POPE Dataset

Download and place under:

```
DATA/POPE/
├── pope_random.json
├── pope_popular.json
└── pope_adversarial.json
```

---

## 📈 Evaluation

### 🎯 Benchmarks

HulluEdit is designed to be evaluated on multiple hallucination benchmarks:

| Benchmark | Status | Description |
|:---|:---:|:---|
| ✅ **POPE** | Available | Polling-based Object Probing Evaluation |
| ⏳ **AMBER** | Coming Soon | Automatic Multi-modal Benchmark |
| ⏳ **Hallu-Bench** | Coming Soon | Comprehensive Hallucination Benchmark |
| ⏳ **CHAIR** | Coming Soon | Caption Hallucination Evaluation |
| ⏳ **MME** | Coming Soon | Multi-Modal Evaluation |
| ⏳ **LLaVA-Bench** | Coming Soon | LLaVA Benchmark |
| ⏳ **MMVet** | Coming Soon | Multi-Modal Visual Reasoning |

> 💡 **Current Status:** **POPE** evaluation is supported. Additional evaluation scripts will be released in future updates!

### 🔬 POPE Evaluation

```bash
# Navigate to project directory
cd /root/Hullu
conda activate hullu

# Run POPE evaluation for LLaVA
bash scripts/run_pope_llava.sh

# Run POPE evaluation for mPLUG-Owl2
bash scripts/run_pope_mplug.sh
```

### 📂 Project Structure

> 💡 **Core Implementation:** The main implementation of HulluEdit is located in the `hulluedit/` directory.

```
HulluEdit/
├── hulluedit/              # 🔥 Core implementation
│   ├── steer.py           # Main weight editing method
│   ├── engines/           # Model engines (LLaVA, MiniGPT-4, mPLUG-Owl2)
│   └── eval/              # Evaluation scripts
├── configs/               # Configuration files
├── scripts/               # Evaluation scripts
├── docs/                  # Documentation & figures
└── DATA/                  # Dataset directory
```

---

## 📝 Citation

```bibtex
@inproceedings{hulluedit2026,
  title={HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models},
  author={Lin, Yangguang and Fang, Quan and Li, Yufei and Sun, Jiachen and Gao, Junyu and Sang, Jitao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 🙏 Acknowledgments

This work is built upon the excellent foundations of [DeCo](https://github.com/zjunlp/Deco) and [AlphaEdit](https://github.com/jianghoucheng/AlphaEdit). We sincerely thank the authors for their valuable contributions to the research community.

---

⭐ **Star us on GitHub** if this project helps you!
