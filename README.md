# 🎯 CLIP-Enhanced Multimodal CTR Prediction

A two-stage deep learning solution for Click-Through Rate (CTR) prediction combining CLIP embeddings with Transformer-DCN architecture for the WWW 2025 Multimodal CTR competition.

## 🔍 Overview

This project implements an advanced CTR prediction system that leverages multimodal features (images + user interaction history) to predict user engagement. The solution is designed for the MicroLens dataset and achieves state-of-the-art performance through a two-stage approach.

**Key Highlights:**
- 🖼️ Vision Transformer embeddings via CLIP
- 🔄 Sequential pattern learning with Transformers
- 🎓 Deep Cross Network v2 for feature interactions
- ⚡ GPU-accelerated training pipeline
- 📊 AUC: 0.8924 on validation set

## 🏗️ Architecture

### Stage 1: CLIP Embedding Generation
- **Model:** `openai/clip-vit-base-patch32`
- **Hardware:** Dual T4 GPUs for parallel inference
- **Process:** 
  - Extracts 512-d semantic vectors from item images
  - Applies PCA to reduce dimensionality to 128-d
  - Preserves rich visual semantics for downstream tasks

### Stage 2: Transformer-DCN Training
```
┌─────────────────────────────────────────────────────────────────┐
│                        Predicted CTR                            │
│                             ▲                                   │
│                             │                                   │
│                    ┌────────┴────────┐                         │
│                    │      MLP        │  Prediction Layer       │
│                    └────────▲────────┘                         │
│                             │                                   │
│                    ┌────────┴────────┐                         │
│       side info ──▶│     DCNv2       │  Feature Iteration     │
│                    └────────▲────────┘                         │
│  ┌──────────────────────────┴──────────────────────────────┐  │
│  │       Sequential Features Learning                       │  │
│  │                                                           │  │
│  │          ┌──────────────────────────────────┐           │  │
│  │          │   Transformer Encoder            │           │  │
│  │          └──────────────▲───────────────────┘           │  │
│  │                         │                        ×N      │  │
│  │          ┌──────────────┴───────────────────┐           │  │
│  │          │   Transformer Encoder            │           │  │
│  │          └──────────────▲───────────────────┘           │  │
│  │                         │                                │  │
│  │          ┌──────────────┴───────────────────┐           │  │
│  │          │  Item Sequence Embeddings        │           │  │
│  │          │  (with padding mask)             │           │  │
│  │          └──────────────────────────────────┘           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│              ┌──────────────┴──────────────┐                   │
│              │                              │                   │
│  ┌───────────┴────────────┐  ┌─────────────┴──────────────┐   │
│  │ Cached Multimodal      │  │ Learnable ID Embeddings    │   │
│  │ Embeddings (frozen)    │  │ (trainable)                │   │
│  │ • CLIP features 128-d  │  │ • User embeddings          │   │
│  │ • Item visual features │  │ • Item ID embeddings       │   │
│  └────────────────────────┘  └────────────────────────────┘   │
│              │                              │                   │
│  ┌───────────┴──────────────────────────────┴──────────────┐   │
│  │              User Interaction History                    │   │
│  │  [Item1] [Item2] [Item3] ... [ItemN] → [Target Item]   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```
**Architecture Components:**
- **Cached Multimodal Embeddings:** Pre-computed CLIP features (frozen)
- **Learnable ID Embeddings:** User and item categorical features
- **Sequential Features Learning:** 
  - Transformer Encoder processes user interaction history
  - Attention mechanism captures temporal patterns
  - Padding mask handles variable-length sequences
- **Feature Interaction:** DCNv2 learns high-order cross features
- **Prediction Layer:** MLP outputs final CTR probability

## ✨ Features

- **Multimodal Learning:** Combines visual (CLIP) and behavioral (user history) signals
- **Memory Efficient:** Caches CLIP embeddings to avoid repeated encoding
- **Scalable:** Handles 3.6M training samples with 91K+ items
- **Flexible:** Supports both image-based and ID-based features
- **Production Ready:** Includes data preprocessing, training, and inference pipelines

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (T4 or better recommended)
- 16GB+ RAM

### Setup
```bash
# Clone the repository
git clone https://github.com/Anas-Tou/MM-CTR-Comp.git
cd clip-enhanced-ctr

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
fuxictr==2.3.7
pandas==2.2.3
scikit-learn==1.4.0
transformers==4.38.0
torch>=2.0.0
pillow
tqdm
numpy
```

## 📊 Dataset

This project uses the **MicroLens-1M** dataset:
```
MicroLens_1M_x1/
├── train.parquet          # 3.6M training samples
├── valid.parquet          # 10K validation samples
├── test.parquet           # 379K test samples
├── item_info.parquet      # Item metadata
└── item_images/           # 91K item images
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

**Data Schema:**
- `user_id`: User identifier
- `item_seq`: Sequence of previously interacted items
- `item_id`: Target item
- `likes_level`, `views_level`: Engagement metrics (categorical)
- `item_tags`: Item category tags (multi-hot encoded)
- `label`: Binary CTR label (0/1)

## 🚀 Usage

### 1. Generate CLIP Embeddings
```python
# Configure paths
IMG_ROOT = "./data/item_images"
ITEM_INFO_PATH = "./data/MicroLens_1M_x1/item_info.parquet"

# Run embedding extraction
python generate_embeddings.py
```

This creates `item_info_task1.parquet` with a new column `item_emb_d128`.

### 2. Train the Model
```bash
python run_expid.py \
    --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 \
    --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c \
    --gpu 0
```

**Training Configuration:**
- Batch size: 1024
- Epochs: 5
- Optimizer: Adam (lr=0.001)
- Early stopping: Patience 2

### 3. Generate Predictions
```bash
python prediction.py \
    --config config/Transformer_DCN_microlens_mmctr_tuner_config_01 \
    --expid Transformer_DCN_MicroLens_1M_x1_001_820c435c \
    --gpu 0
```

Output: `submission/prediction.csv`

### Quick Start (End-to-End)

Run the complete pipeline in the Jupyter notebook:
```bash
jupyter notebook NoteBook.ipynb
```

## 📈 Model Performance

| Metric | Validation | Test |
|--------|-----------|------|
| **AUC** | 0.8924 | TBD |
| **LogLoss** | 1.4100 | TBD |

**Training Progress:**

| Epoch | Train Loss | Val AUC | Status |
|-------|-----------|---------|--------|
| 1 | 0.1590 | 0.8431 | ✓ |
| 2 | 0.0586 | 0.8659 | ✓ |
| 3 | 0.0414 | 0.8506 | Early Stop |
| 4 | 0.0288 | 0.8908 | ✓ |
| 5 | 0.0261 | **0.8924** | Best |

### Comp Result
<img width="1627" height="75" alt="image" src="https://github.com/user-attachments/assets/977bbd7b-7c5d-4de5-9890-c66085b1bb6d" />

### Training Time
- **CLIP Embedding Generation:** ~11 minutes (91K images, Dual T4)
- **Model Training:** ~16 minutes per epoch
- **Total Training Time:** ~1.5 hours

## 🔧 Customization

### Adjust Model Hyperparameters

Edit `config/Transformer_DCN_microlens_mmctr_tuner_config_01/model_config.yaml`:
```yaml
# Key parameters
embedding_dim: 32              # ID embedding size
transformer_layers: 1          # Number of transformer blocks
num_heads: 1                   # Attention heads
dim_feedforward: 128           # FFN hidden size
dcn_cross_layers: 3            # DCN depth
dcn_hidden_units: [512, 256]   # DCN layer dimensions
mlp_hidden_units: [32]         # Final MLP layers
learning_rate: 0.001           # Initial LR
batch_size: 1024               # Training batch size
net_dropout: 0.1               # Dropout rate
transformer_dropout: 0.1       # Transformer dropout
first_k_cols: 8                # Number of recent items to focus on
```
## 🙏 Acknowledgments

- **FuxiCTR** framework for CTR model implementations
- **OpenAI CLIP** for powerful vision-language embeddings
- **Team Momo** for the Transformer-DCN architecture inspiration
- Course instructors for the MicroLens dataset and guidance
- **Hugging Face** for the transformers library

## 📚 References

1. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
2. Wang, R., et al. (2021). "DCN V2: Improved Deep & Cross Network." SIGIR.
3. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
4. Zhou, G., et al. (2018). "Deep Interest Network for Click-Through Rate Prediction." KDD.
##

</div>
