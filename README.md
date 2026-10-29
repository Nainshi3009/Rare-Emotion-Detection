# Rare Emotion Detection in Text Using Continuous Affective Quality Gates and Multi-Task RoBERTa

## Abstract and Overview

Fine-grained emotion classification models trained on benchmark corpora such as GoEmotions encounter substantial performance degradation on rare emotional categories (including grief, pride, relief, and nervousness). Because these categories comprise less than 0.5% of the total dataset annotations, conventional discriminative classifiers optimize aggregate loss by predicting the majority classes, yielding zero recall on the minority distribution at standard decision boundaries.

While generative data augmentation utilizing Large Language Models provides a theoretical mechanism for sample expansion, unconstrained synthetic text is susceptible to label hallucination and affective drift. This repository implements a multi-task learning architecture based on RoBERTa-base, coupling 28-class multi-label classification with continuous Valence-Arousal-Dominance (VAD) regression derived from the NRC-VAD Lexicon. Furthermore, a continuous Euclidean distance quality gate is established to filter candidate synthetic examples prior to model ingestion.

---

## Key Methodological Components

### 1. Continuous Affective Quality Gate
Synthetic training candidates generated for rare emotion categories are assigned continuous sentence-level Valence, Arousal, and Dominance (VAD) coordinates via lexical aggregation against the NRC-VAD Lexicon. Each candidate vector is evaluated against the expected centroid coordinate of the target emotional state. Candidates exceeding the Euclidean distance threshold ($\tau = 0.35$) are discarded, preventing affective drift and label dilution.

### 2. Multi-Task Architecture
The model utilizes a shared RoBERTa-base encoder coupled with two output heads:
- **Classification Head**: Computes 28 independent binary logits optimized via Binary Cross-Entropy with Logits loss ($\mathcal{L}_{\text{BCE}}$).
- **Regression Head**: Predicts continuous 3-dimensional affective scores (Valence, Arousal, Dominance) optimized via Mean Squared Error loss ($\mathcal{L}_{\text{MSE}}$).
- **Joint Objective**: Optimized end-to-end via $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda \cdot \mathcal{L}_{\text{MSE}}$, where $\lambda = 0.5$.

---

## Experimental Evaluation

The methodology is benchmarked across four controlled experimental conditions evaluated on identical held-out test partitions:

1. **Condition A (Gold Only)**: Baseline model trained exclusively on authentic GoEmotions training rows.
2. **Condition B (Standard Oversampling)**: Baseline model trained on authentic rows with minority classes duplicated to target frequency.
3. **Condition C (Unfiltered Synthetic)**: Model trained on authentic data augmented with unconstrained synthetic candidates.
4. **Condition D (VAD-Filtered Synthetic)**: Model trained on authentic data augmented with candidates validated through the continuous VAD quality gate.

### Summary of Experimental Results

- **Continuous Dimensional Error Reduction**: Auxiliary continuous regression error decreased substantially under synthetic augmentation. Mean Absolute Error (MAE) for Valence decreased by 24.0% (from 0.1970 in Gold Only to 0.1498 in Condition D), and Arousal MAE decreased by 16.3% (from 0.1061 to 0.0888).
- **Contextual Generalization**: Overall test weighted F1 improved from 2.30% in the baseline to 3.01% in synthetic conditions (a 30.8% relative gain).
- **Quality Gate Sensitivity**: The NRC-VAD distance gate rejected 12.9% of candidate texts exhibiting affective divergence from the target emotion centroid.
- **Calibration Analysis**: Expected Calibration Error (ECE) decreased from 0.4377 to 0.4133, and Brier score decreased from 0.2353 to 0.2130, confirming improved probability calibration.

---

## Repository Structure

```text
├── configs/                  Model hyperparameter and training formulation specifications
│   ├── gold_only_baseline.json
│   ├── model_resource_efficient.json
│   └── model_training_formulation.json
├── docs/                     Detailed methodology and loss formulation audits
├── reports/                  Consolidated experimental outputs and evaluation figures
│   ├── figures/              Generated comparison visualizations
│   ├── calibration_and_ece_analysis.json
│   └── experiment_comparison.json
├── src/                      Core model architecture, dataset, and training modules
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── model.py
│   ├── training.py
│   └── vad_targets.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation and Environment Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher

### Setup

```bash
git clone https://github.com/nagratana/Rare-Emotion-Detection.git
cd Rare-Emotion-Detection
pip install -r requirements.txt
```

---

## Dataset and Lexicon References

- **GoEmotions**: Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions," Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
- **NRC VAD Lexicon**: Mohammad, S. M., "Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words," Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2018.
- **RoBERTa**: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.
