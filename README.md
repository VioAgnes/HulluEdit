# HulluEdit 

HulluEdit is a hallucination-mitigation engine for multimodal large language models. It estimates a visual evidence subspace together with an anti-prior subspace and applies a closed-form edit to the hidden states, preserving description richness while reducing factual errors. The current release focuses on LLaVA-1.5-7B and ships an evaluation pipeline aligned with POPE metrics.

## Model Overview 🌐

  ![Model diagram](docs/model.png)

## Environment Setup 🛠️

- 🐍 Create a Python 3.10+ environment (Conda is recommended):
  ```bash
  conda create -n hullu python=3.10
  conda activate hullu
  ```
  
- 📦 Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Data Preparation 📦

Download the COCO 2014 validation split and arrange the files as follows (adjust the root path to your storage location):
```
DATA/
├── annotations/
│   ├── instances_val2014.json
│   └── captions_val2014.json
└── val2014/
    ├── COCO_val2014_000000000042.jpg
    └── ...
```
- 🔧 Set `COCO_ANNOTATIONS` or other environment variables before running scripts if your layout differs from the defaults.

## Evaluate (LLaVA) 🔍

```bash
cd Hulluedit
conda activate hullu
bash scripts/pope_llava.sh
```

