# Pet Breed Identifier

> **AI module that determines whether two pet photos share the same breed.**  
> Built with CLIP embeddings + zero-shot classification. Supports 102 dog & cat breeds.

---

## Table of Contents
1. [Quick Start](#-quick-start)
2. [Approach & Architecture](#-approach--architecture)
3. [Project Structure](#-project-structure)
4. [Usage](#-usage)
5. [Evaluation & Limitations](#-evaluation--limitations)
6. [Bonus Features](#-bonus-features)

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd pet_breed_identifier

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies  (~2 min, downloads CLIP weights on first run)
pip install -r requirements.txt

# 4a. Launch the Streamlit web app
streamlit run app.py

# 4b. OR run the CLI
python cli_demo.py compare dog1.jpg dog2.jpg
python cli_demo.py identify cat.jpg
```

> **First run** downloads the CLIP ViT-B/32 weights (~340 MB) from HuggingFace.  
> Subsequent runs are instant because the model is cached locally.

---

## Approach & Architecture

### Model — CLIP (ViT-B/32)

[CLIP](https://openai.com/research/clip) (Contrastive Language–Image Pre-training) by OpenAI
maps both images and text into a shared 512-dimensional embedding space.  
This makes it ideal for **zero-shot breed classification** without any task-specific fine-tuning.

```
Image ──► CLIP Image Encoder ──► 512-d unit vector (image embedding)
Text  ──► CLIP Text  Encoder ──► 512-d unit vector (text  embedding)
```

### Step-by-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PawMatch Pipeline                            │
│                                                                     │
│  Image 1 ──► CLIP Encoder ──► Embedding E1 ──┐                     │
│                                               ├─► Cosine Similarity │
│  Image 2 ──► CLIP Encoder ──► Embedding E2 ──┘    (primary signal) │
│                                                                     │
│  E1 ──► @ Text Embeddings ──► Softmax ──► Breed 1 + confidence     │
│  E2 ──► @ Text Embeddings ──► Softmax ──► Breed 2 + confidence     │
│                                                                     │
│  Decision: sim ≥ 0.82  OR  Breed1 == Breed2  →  SAME BREED        │
│            otherwise                          →  DIFFERENT BREED   │
└─────────────────────────────────────────────────────────────────────┘
```

### Zero-Shot Breed Classification

At initialisation, text embeddings are pre-computed for all 102 breed prompts:

```python
"a photo of a Golden Retriever"
"a photo of a Siamese"
# … 100 more
```

### Same/Different Decision Logic

| Cosine Similarity | Breed Labels Agree | Verdict        | Confidence |
|-------------------|--------------------|----------------|------------|
| ≥ 0.82            | ✅                 | **Same**       | High       |
| ≥ 0.82            | ❌                 | **Same**       | Moderate   |
| < 0.82            | ✅                 | **Same**       | Moderate   |
| < 0.82            | ❌                 | **Different**  | High       |

---

## 📁 Project Structure

```
pawmatch-pet-breed-identifier/
│
├── breed_identifier.py   # ← Core ML module (BreedIdentifier class)
├── app.py                # ← Streamlit web UI
├── cli_demo.py           # ← CLI demo script
├── demo_notebook.ipynb   # ← Jupyter notebook walkthrough
├── requirements.txt      # ← Python dependencies
└── README.md             # ← This file
```

---

## 🖥 Usage

### Python API

```python
from PIL import Image
from breed_identifier import BreedIdentifier

identifier = BreedIdentifier()  # loads model once

img1 = Image.open("dog_a.jpg")
img2 = Image.open("dog_b.jpg")

result = identifier.compare(img1, img2)

print(result.verdict)
# Same Breed — both appear to be Golden Retriever (confidence: 91.3%)

print(result.same_breed)       # True
print(result.confidence)       # 0.913
print(result.similarity_score) # 0.8947
```

### CLI

```bash
# Compare two images
python cli_demo.py compare dog1.jpg dog2.jpg

# Identify single image
python cli_demo.py identify cat.jpg
```

### Web App (Streamlit)

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## 📊 Evaluation & Limitations

### What Works Well
- **Cross-lighting / angle invariance** — CLIP embeddings are robust to photo conditions.
- **Both dogs and cats** — 61 dog breeds + 41 cat breeds, no retraining needed.
- **No labelled data required** — pure zero-shot; can add new breeds instantly.
- **Top-3 suggestions** — partial breed mixes and ambiguous photos still surface useful candidates.

### Limitations

| Limitation | Impact | Potential Fix |
|---|---|---|
| Zero-shot accuracy ~70-80% vs fine-tuned ~90%+ | Occasional misclassification of similar breeds | Fine-tune on Oxford Pets / Stanford Dogs |
| Threshold (0.82) is heuristic | Edge cases near the boundary | Calibrate on labelled pairs dataset |
| 102 breeds only | Rare/exotic breeds default to closest known breed | Expand breed list or use open-vocabulary CLIP |

---

## Bonus Features

-  **Top-3 breed suggestions** (not just binary output)
-  **Multi-species support** (dogs + cats in the same model)
-  **Confidence score** (calibrated from similarity + label agreement)
-  **Streamlit web app** with dark theme UI
-  **CLI demo** with coloured terminal output
-  **Modular design** — `BreedIdentifier` is fully importable as a library

---

## Dataset & Model Credits

| Resource | Source |
|---|---|
| CLIP ViT-B/32 | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| Dog breed list | [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) |
| Cat breed list | [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) |

---

*Built as a 2-day assignment — focus on working pipeline, clear architecture, and explainability.*
