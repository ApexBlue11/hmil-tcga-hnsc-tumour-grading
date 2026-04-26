# H-MIL Tumour Grading — TCGA-HNSC

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/ApexBlue720/TumourGrading_Model)
[![JAX](https://img.shields.io/badge/JAX-Flax-orange)](https://github.com/google/flax)
[![TPU](https://img.shields.io/badge/TPU-v5e--8-green)](https://cloud.google.com/tpu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hierarchical Multiple Instance Learning (H-MIL) model for automated tumour grading of Head and Neck Squamous Cell Carcinoma (HNSC) from Whole Slide Images, trained on TCGA-HNSC. Implemented in JAX/Flax and trained on TPU v5e-8.

**Ensemble QWK: 0.6683** across a 3-seed ensemble (seeds 42, 7, 123).

---

## Overview

Whole Slide Images (WSIs) are gigapixel-scale — too large to process as a single input. This project treats each slide as a *bag* of patch embeddings and uses a two-level (hierarchical) attention transformer to aggregate local patch context into region representations, then global region context into a slide-level grade prediction.

Grade classes follow TCGA convention:
- **G1** — Well differentiated  
- **G2** — Moderately differentiated  
- **G3** — Poorly differentiated  

---

## Architecture

Patch embeddings (1024-dim, from UNI ViT-L) are fed into a two-stage transformer:

```
[N patches × 1024] 
       ↓  Linear projection → hidden_dim
[Local Transformer]   ← Pre-LayerNorm + residual, 4 heads
       ↓  CLS token pooling per region
[Global Transformer]  ← Pre-LayerNorm + residual, 4 heads  
       ↓  CLS token pooling
[Classifier Head]     → 3-class logits (G1 / G2 / G3)
```

**Key design decisions:**
- **Pre-LayerNorm + residual connections** on both transformer stages — eliminates training instability (epoch-level F1 swings seen in Post-LN baseline)
- **Pseudo-bag subsampling** (DTFD strategy, cap 10k patches) — acts as stochastic data augmentation, prevents overfitting on slide size
- **Ordinal Cross-Entropy loss** = weighted CE + α × (expected grade distance)² — penalises grade-skipping predictions more heavily than adjacent misclassifications
- **Layer-wise learning rates** — classifier > global transformer > local transformer > projection layer
- **TPU data parallelism** across 8 cores via `pmap`

---

## Training

**Hyperparameter optimisation** (`training/hmil_optuna_hpo.ipynb`):
- Optuna TPE sampler, warm-started from 2 manually seeded trials from a prior 23-trial run
- 12 total trials, QWK as primary metric
- Search space: learning rate, dropout, region size, batch size, ordinal loss weight α
- Best trial: QWK = 0.5861, F1 = 0.6296

**Final ensemble training** (`training/hmil_final_ensemble.ipynb`):
- Fixed hyperparameters from best Optuna trial
- 3 independent seeds (42, 7, 123), 60 epochs each, patience = 10
- Ensemble by averaging softmax probabilities across seeds
- Deterministic validation collation across all seeds for aligned ensemble

| Metric | Value |
|--------|-------|
| Ensemble QWK | **0.6683** |
| Primary data | TCGA-HNSC patch embeddings (UNI ViT-L, 1024-dim) |
| Hardware | Kaggle TPU v5e-8 |
| Framework | JAX + Flax + Optax |

---

## Inference Demo

A Streamlit app is hosted on HuggingFace Spaces for inference using pre-extracted embeddings.  
The app runs embedding-only H-MIL inference (no UNI patch extraction) to stay within free-tier compute limits.

👉 **[Try the live demo](https://huggingface.co/spaces/ApexBlue720/TumourGrading_Model)**

App source is in the `app/` directory.

---

## Repository Structure

```
.
├── training/
│   ├── hmil_optuna_hpo.ipynb        # Optuna HPO — 12 trials on TPU v5e-8
│   ├── hmil_final_ensemble.ipynb    # 3-seed ensemble training + confusion matrix
│   └── dataset_utils.py             # SlideDataset, collation functions
│
├── app/
│   ├── streamlit_app.py             # Inference UI (HuggingFace Spaces)
│   ├── requirements.txt
│   └── Dockerfile
│
└── README.md
```

---

## Setup

These notebooks are designed for **Kaggle TPU v5e-8** (free tier). Running locally requires a JAX-compatible accelerator.

```bash
pip install jax jaxlib flax optax optuna scikit-learn
```

Data dependencies:
- `tcga-hnsc-pt-embeddings` — pre-extracted UNI patch embeddings (`.pt` files per slide)
- `tcga_hnsc_labels.csv` — TCGA manifest with grade labels

Both are available as Kaggle datasets under the `apexblue` namespace.

---

## References

## Acknowledgements & Citations

This project utilizes datasets, architectural frameworks, and foundational models developed by the broader research community. If you find this work useful, please ensure you also attribute the original authors below:

**Foundational Pathology Model (UNI)**
* Chen, R.J., Ding, T., Lu, M.Y., Williamson, D.F.K., et al. (2024). Towards a general-purpose foundation model for computational pathology. *Nature Medicine*. [https://doi.org/10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3)

**Histopathological Segmentation (MIL)**
* Lerousseau, M., Vakalopoulou, M., Classe, M., et al. (2020). Weakly supervised multiple instance learning histopathological tumor segmentation. In *Medical Image Computing and Computer Assisted Intervention–MICCAI 2020* (pp. 470-479). Springer. [https://doi.org/10.48550/arXiv.2004.05024](https://doi.org/10.48550/arXiv.2004.05024)

**Dataset (TCGA-HNSC)**
* Cancer Genome Atlas Network. (2015). Comprehensive genomic characterization of head and neck squamous cell carcinomas. *Nature, 517*(7536), 576-582. [https://doi.org/10.1038/nature14129](https://doi.org/10.1038/nature14129)

**Core Architectures utilized by UNI (ViT & DINOv2)**
* Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.
* Oquab, M., Darcet, T., Moutakanni, T., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research*.

### BibTeX

<details>
<summary>Click to expand BibTeX citations</summary>

```bibtex
@article{chen2024uni,
  title={Towards a General-Purpose Foundation Model for Computational Pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}

@inproceedings{lerousseau2020weakly,
  title={Weakly supervised multiple instance learning histopathological tumor segmentation},
  author={Lerousseau, Marvin and Vakalopoulou, Maria and Classe, Marion and Adam, Julien and Battistella, Enzo and Carr{\'e}, Alexandre and Estienne, Th{\'e}o and Henry, Th{\'e}ophraste and Deutsch, Eric and Paragios, Nikos},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
  pages={470--479},
  year={2020},
  organization={Springer}
}

@article{cancergenome2015comprehensive,
  title={Comprehensive genomic characterization of head and neck squamous cell carcinomas},
  author={{Cancer Genome Atlas Network}},
  journal={Nature},
  volume={517},
  number={7536},
  pages={576--582},
  year={2015},
  publisher={Nature Publishing Group}
}

@inproceedings{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={[https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)}
}

@article{oquab2024dinov2,
  title={{DINO}v2: Learning Robust Visual Features without Supervision},
  author={Maxime Oquab and Timoth{\'e}e Darcet and Th{\'e}o Moutakanni and Huy V. Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Herve Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
  journal={Transactions on Machine Learning Research},
  year={2024},
  url={[https://openreview.net/forum?id=a68SUt6zFt](https://openreview.net/forum?id=a68SUt6zFt)}
}
