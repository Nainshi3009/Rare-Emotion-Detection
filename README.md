Rare Emotion Detection in Text Using Continuous Affective Quality Gates and Multi-Task RoBERTa
Author: Nainshi  
Repository: https://github.com/Nainshi3009/Rare-Emotion-Detection  
Dataset: GoEmotions · Lexicon: NRC-VAD (2018) · Backbone: `roberta-base`
---
Abstract
Fine-grained emotion classification models trained on large benchmark corpora such as GoEmotions encounter severe performance degradation on rare emotional categories — specifically grief, pride, relief, and nervousness — which together comprise less than 0.5 % of total dataset annotations. Standard discriminative classifiers optimise aggregate loss by predicting majority classes, yielding near-zero recall on minority labels at standard decision boundaries.
This project implements a multi-task learning architecture grounded in RoBERTa-base, coupling 28-class multi-label classification with continuous Valence–Arousal–Dominance (VAD) regression derived from the NRC-VAD Lexicon. Synthetic text candidates generated for under-represented emotions are subjected to a continuous Euclidean-distance quality gate before model ingestion, preventing affective drift and label dilution. The pipeline is benchmarked across four controlled experimental conditions (A–D), of which Condition A (Gold-Only Baseline) has been fully completed.
---
Table of Contents
Problem Statement and Objectives
Dataset and Technology Stack
System Architecture and Methodology
NRC-VAD Filtering
Multi-Task RoBERTa Model
Experimental Design (A–D)
Verified Results
Limitations and Pending Work
Repository Structure
Installation and Environment Setup
Execution Instructions
Running Tests
Secret Management
Research References
Future Work
---
Problem Statement and Objectives
Problem
GoEmotions is the largest publicly available fine-grained emotion dataset (58,009 Reddit comments, 28 classes). Yet the distribution is severely long-tailed: four emotions account for a disproportionately small share of training annotations:
Emotion	Approx. GoEmotions train count
grief	~115
pride	~177
relief	~198
nervousness	~182
A model trained on these data without intervention will achieve close-to-zero recall for these categories while maintaining competitive macro/weighted metrics driven by high-frequency classes (admiration, approval, gratitude, etc.).
Objectives
Establish a reproducible Gold-Only baseline (Condition A) that faithfully reflects the difficulty of rare-class detection.
Implement NRC-VAD lexicon-derived pseudo-targets to provide a continuous affective supervision signal as a regulariser via multi-task learning.
Design a VAD quality gate (Euclidean distance threshold τ = 0.35) to filter Gemini-generated synthetic candidates for target emotions.
Compare four training data strategies (Conditions A–D) on identical held-out test partitions using Macro-F1, Weighted F1, and per-rare-class F1 as primary metrics.
---
Dataset and Technology Stack
Dataset
GoEmotions (Demszky et al., ACL 2020) — `google-research-datasets/go_emotions` via Hugging Face Datasets
Train: 43,410 examples · Validation: 5,426 · Test: 5,427
28 emotion classes (multi-label, one or more per example)
Affective Lexicon
NRC-VAD Lexicon (Mohammad, ACL 2018) — word-level Valence, Arousal, Dominance scores in [0, 1]
Used to compute sentence-level pseudo-VAD targets by averaging word-level scores over tokens present in the lexicon.
VAD targets are lexicon-derived pseudo-targets, not human-annotated sentence-level VAD ground truth.
Synthetic Data
Google Gemini API (`gemini-3.8-flash`) — used to generate candidate text examples for the four target rare emotions.
Total synthetic examples generated: 30 (across two batches)
VAD filter pass rate: 30/30 (100 %) — all generated candidates passed the Euclidean-distance threshold.
Technology Stack
Component	Library / Version
Backbone	`roberta-base` via `transformers ≥ 4.30`
Training framework	`torch ≥ 2.0`
Dataset loading	`datasets ≥ 2.14`, `pyarrow ≥ 12.0`
Data processing	`pandas ≥ 2.0`, `numpy ≥ 1.24`, `scikit-learn ≥ 1.3`
API client	`google-genai`, `httpx`
Environment	`python-dotenv ≥ 1.0`
Resource monitoring	`psutil ≥ 5.9`
Python	3.10 or higher
---
System Architecture and Methodology
Overview
```
GoEmotions Dataset
        │
        ▼
 data_preparation.py ──────────────────────────────────────────────────────────────┐
  • Load & validate splits (hash-verified)                                          │
  • Build NRC-VAD pseudo-targets per example                                        │
  • (Condition B) Duplicate rare-class rows to target frequency (164)              │
  • (Conditions C/D) Merge VAD-filtered synthetic candidates                        │
        │                                                                           │
        ▼                                                                           │
 GoEmotionsVADDataset (dataset.py)                                                  │
  • Tokenise with RoBERTa tokeniser (max_length 128)                               │
  • Produce input_ids, attention_mask, labels (28-dim binary), vad_targets (3-dim) │
  • vad_mask: True where NRC-VAD coverage ≥ 1 token                                │
        │                                                                           │
        ▼                                                                           │
 RobertaEmotionVADModel (model.py)                                                  │
  ┌──────────────────────────────────────────────────────────────────────────────┐  │
  │  roberta-base encoder (12 transformer layers, hidden=768)                    │  │
  │      │                                                                       │  │
  │      ├── emotion_head (Linear 768 → 28)   → BCEWithLogitsLoss               │  │
  │      └── vad_head    (Linear 768 → 3)     → Masked MSE                       │  │
  │                                                                              │  │
  │  total_loss = L_BCE + λ × masked_VAD_MSE   (λ = 0.5)                        │  │
  └──────────────────────────────────────────────────────────────────────────────┘  │
        │                                                                           │
        ▼                                                                           │
 training.py                                                                        │
  • AdamW, linear warmup (10 %), gradient accumulation (eff. batch = 8)            │
  • Best-checkpoint selection on validation Macro-F1                                │
  • Full recovery-checkpoint support (epoch, step, RNG state)                      │
  • Test evaluation: per-class P/R/F1, Macro-F1, Weighted-F1, Rare-emotion-F1     │
        │                                                                           │
        ▼                                                                           │
 outputs/experiments/{condition}/                                                   │
  • metrics.json, per_class_metrics.json, rare_emotion_metrics.json               │
  • best_checkpoint.pt                                                              │
```
Loss Formulation
The training objective was determined after a methodological audit documented in `docs/multilabel_loss_audit.md`:
$$\mathcal{L}{\text{total}} = \underbrace{\mathcal{L}{\text{BCE}}}{\text{28-class multi-label}} + \lambda \cdot \underbrace{\mathcal{L}{\text{VAD-MSE}}}_{\text{masked 3-dim regression}}$$
Classification loss: `BCEWithLogitsLoss` applied to the raw 28-dimensional multi-hot binary targets (0 or 1 per label). No softmax or target normalisation.
VAD regression loss: Mean Squared Error over the 3-dimensional (Valence, Arousal, Dominance) head output, masked so only examples with at least one NRC-VAD token contribute.
λ = 0.5 (fixed for all conditions).
Threshold: Fixed at 0.5 for sigmoid-based classification during evaluation.
---
NRC-VAD Filtering
Synthetic candidate sentences are generated via the Gemini API for each target emotion. Before any candidate is admitted into training data, a quality gate is applied:
Gate Algorithm (`src/vad_targets.py`, `scripts/generate_synthetic_completion.py`)
Lexical VAD Scoring — For each candidate sentence, compute sentence-level Valence, Arousal, and Dominance by averaging NRC-VAD scores over all tokens present in the lexicon.
Centroid Reference — Retrieve the expected VAD centroid for the target emotion (derived from GoEmotions training examples for that label).
Euclidean Distance — Compute $d = |v_{\text{candidate}} - v_{\text{centroid}}|_2$ in the 3-dimensional VAD space.
Gate Decision — Accept the candidate if $d \leq \tau = 0.35$; discard otherwise.
Filter Results (Verified)
Batch	Candidates generated	Passed gate	Rejected
Batch 1	20	20	0
Batch 2	10	10	0
Total	30	30	0
> **Note:** A 100 % pass rate is consistent with careful prompt engineering that constrains Gemini to generate text with emotionally coherent lexical signals. It does not imply that the synthetic examples are indistinguishable from gold training data, nor that augmentation improves classification performance (Conditions C/D not yet completed).
---
Multi-Task RoBERTa Model
Architecture (`src/model.py`)
```python
class RobertaEmotionVADModel(nn.Module):
    encoder:       RoBERTa-base (pretrained)     # 12 layers, hidden=768
    emotion_head:  Linear(768, 28)               # multi-label classification
    vad_head:      Linear(768, 3)                # VAD regression
```
Freeze strategies (configurable via `freeze_strategy`):
Strategy	Trainable parameters	% of total
`full`	All 124M	100 %
`last_2_layers`	Layers 10–11 + heads	~11.44 % (≈14.2M)
The resource-efficient configuration (`configs/model_resource_efficient.json`) uses `last_2_layers` to enable training on CPU hardware with limited memory.
---
Experimental Design (A–D)
All four conditions share the same model architecture, loss formulation, hyperparameters, and held-out test split. They differ only in the training data composition:
Condition	Label	Training data
A	Gold Only	Original GoEmotions train split (43,410 rows). No augmentation.
B	Standard Oversampling	Gold data + minority-class rows duplicated until each rare emotion reaches the target frequency (164 examples).
C	Unfiltered Synthetic	Gold data + raw Gemini-generated synthetic candidates (no VAD gate).
D	VAD-Filtered Synthetic	Gold data + VAD-gate-passed synthetic candidates only.
Hyperparameters (all conditions)
Parameter	Value
`model_name`	`roberta-base`
`max_length`	128
`batch_size`	2
`gradient_accumulation_steps`	4
`effective_batch_size`	8
`epochs`	1
`learning_rate`	2e-5
`weight_decay`	0.01
`warmup_ratio`	0.10
`lambda_vad`	0.5
`classification_threshold`	0.5
`seed`	20261008
`device`	cpu
`optimizer`	AdamW
---
Verified Results
> **Important:** Only results from Condition A (Gold-Only Baseline) have been verified by running the full training pipeline on the complete GoEmotions test split. Conditions B, C, and D are not yet complete. No results from the reference repository are reproduced here.
Condition A — Gold-Only Baseline (Verified)
Metric	Test value
Macro-F1	22.84 %
Weighted-F1	41.32 %
> Rare-class per-label F1 values (grief, pride, relief, nervousness) and calibration metrics (ECE, Brier score) are pending per-class metric extraction from the saved checkpoint.
Conditions B, C, D — Status
Condition	Status
B — Standard Oversampling	⏳ Not yet completed
C — Unfiltered Synthetic	⏳ Not yet completed
D — VAD-Filtered Synthetic	⏳ Not yet completed
Calibration / ECE analysis	⏳ Pending
Generalisation analysis	⏳ Pending
Results will be added to `reports/` upon completion.
---
Limitations and Pending Work
Area	Description
Rare-class recall	With 28-class BCE and a fixed 0.5 threshold, the model may still predict zero positives for rare emotions on the test set; per-class F1 for grief/pride/relief/nervousness may remain at 0.0 even with a competitive Macro-F1.
VAD targets are pseudo-targets	NRC-VAD scores are word-level lexical scores averaged at the sentence level. They are not human-annotated sentence-level VAD ground truth and may be noisy or missing for out-of-vocabulary tokens.
Small synthetic set	Only 30 synthetic examples were generated across two batches. The impact of such a small augmentation set on classification performance is expected to be limited.
CPU training constraint	Training on CPU with a batch size of 2 is extremely slow for full GoEmotions training. All estimates in `reports/training_throughput_benchmark.json` are forward-pass-only.
Threshold tuning	The 0.5 sigmoid threshold is fixed throughout. Threshold calibration per-class (especially for rare emotions) is a pending methodological step.
Conditions B–D	Not completed. Comparative analysis across conditions is not yet possible.
Class weighting	No class weighting is applied in the baseline. Whether weighted loss improves rare-class recall is an open research question for future conditions.
---
Repository Structure
```
Rare-Emotion-Detection/
│
├── .env.example                          # Secret placeholder — copy to .env and fill in
├── .gitignore                            # Excludes .env, data/, models/, outputs/, etc.
│
├── 01_load_goemotions.py                 # EDA: load and inspect GoEmotions via HF Datasets
├── 02_analyze_emotions.py                # EDA: count emotion distribution in training split
├── 03_plot_distribution.py               # EDA: bar chart of emotion frequencies → PNG
├── goemotions_distribution.png           # Saved distribution chart
│
├── generate_synthetic_candidates.py      # Batch 1: Gemini API synthetic generation (20 ex.)
├── generate_synthetic_candidates_batch2.py  # Batch 2: additional 10 examples
├── test_gemini_connection.py             # Verifies Gemini API key and connectivity
│
├── configs/
│   ├── gold_only_baseline.json           # Condition A training hyperparameters
│   ├── model_resource_efficient.json     # last_2_layers freeze config for CPU training
│   └── model_training_formulation.json   # Loss formulation decision record
│
├── docs/
│   └── multilabel_loss_audit.md          # BCE vs CE loss audit and formulation justification
│
├── reports/
│   ├── experiment_readiness.json         # Pre-run readiness check output
│   ├── training_runner_audit.json        # Dry-run audit of training runner
│   └── training_throughput_benchmark.json  # Forward-pass timing estimates (CPU)
│
├── scripts/
│   ├── audit_multilabel_loss.py          # Generates docs/multilabel_loss_audit.md
│   ├── audit_training_runner.py          # Dry-run audit: verifies training pipeline
│   ├── benchmark_training_throughput.py  # CPU forward-pass throughput timing
│   ├── check_experiment_readiness.py     # Validates data, configs, and synthetic files
│   ├── estimate_baseline_cost.py         # Estimates wall-clock time for full training
│   ├── estimate_freezing_strategies.py   # Compares trainable params per freeze strategy
│   ├── generate_synthetic_completion.py  # VAD-aware Gemini synthetic generation script
│   ├── inspect_ml_datasets.py            # Validates prepared Parquet splits
│   ├── inspect_ml_environment.py         # Checks library versions and hardware
│   ├── inspect_model_pipeline.py         # End-to-end forward pass smoke test
│   ├── prepare_vad_targets.py            # Builds NRC-VAD pseudo-targets for all splits
│   ├── run_gold_only.py                  # ▶ Full Condition A training + evaluation
│   ├── run_standard_oversampling.py      # ▶ Full Condition B training + evaluation
│   ├── sanity_train.py                   # 2-step training smoke test (not a real result)
│   └── verify_freeze_configuration.py    # Verifies last_2_layers freeze policy
│
├── src/
│   ├── data_preparation.py               # Dataset loading, VAD merging, oversampling utils
│   ├── dataset.py                        # GoEmotionsVADDataset (PyTorch Dataset)
│   ├── model.py                          # RobertaEmotionVADModel (classification + VAD head)
│   ├── training.py                       # Training loop, evaluation, checkpoint management
│   └── vad_targets.py                    # NRC-VAD lexicon loading and target computation
│
├── tests/
│   └── test_training_resume.py           # Unit tests for checkpoint save/load and RNG restore
│
└── requirements.txt                      # Python dependencies
```
---
Installation and Environment Setup
Prerequisites
Python 3.10 or higher
A UNIX-like terminal or Windows PowerShell
GoEmotions dataset cached locally by Hugging Face Datasets (required before running experiments; download triggers automatically on first `load_dataset` call with network access)
NRC-VAD Lexicon file placed at `data/NRC-VAD-Lexicon.txt` (download from Mohammad 2018)
1. Clone the Repository
```bash
git clone https://github.com/Nainshi3009/Rare-Emotion-Detection.git
cd Rare-Emotion-Detection
```
2. Create a Virtual Environment
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS / Linux
source .venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Configure Secrets
```bash
cp .env.example .env
# Open .env and set your Gemini API key:
#   GEMINI_API_KEY=your_actual_key_here
```
> **Security:** `.env` is excluded from Git via `.gitignore`. Never commit your API key.
5. Verify Environment and Gemini Connection (Optional)
```bash
python scripts/inspect_ml_environment.py
python test_gemini_connection.py
```
---
Execution Instructions
Step 1 — Prepare VAD Targets
This must be run once after the GoEmotions dataset is cached:
```bash
python scripts/prepare_vad_targets.py
```
Outputs prepared Parquet splits to `data/processed/goemotions_vad_v1/`.
Step 2 — Inspect Prepared Data
```bash
python scripts/inspect_ml_datasets.py
```
Step 3 — Run Readiness Check
```bash
python scripts/check_experiment_readiness.py
```
Step 4 — (Optional) Sanity Train (Smoke Test, NOT a result)
```bash
python scripts/sanity_train.py
```
This runs exactly 2 training steps on a 500-row subset. Output values are not experimental results.
Step 5 — Condition A: Gold-Only Baseline
```bash
python scripts/run_gold_only.py
```
Outputs are written to `outputs/experiments/A_gold_only/` (UTC-stamped subdirectory).  
Metrics are saved to `reports/gold_only_baseline/metrics.json`.
Step 6 — Condition B: Standard Oversampling (Pending)
```bash
python scripts/run_standard_oversampling.py
```
Verify Freeze Configuration
```bash
python scripts/verify_freeze_configuration.py
```
---
Running Tests
```bash
python -m pytest tests/ -v
```
The test suite in `tests/test_training_resume.py` covers:
Checkpoint save and load round-trip correctness
RNG state capture and restore (CPU and CUDA)
Recovery checkpoint rejection on config or data mismatch
> **Note:** Tests require PyTorch. No network access or GPU is required; all tests run on CPU.
---
Secret Management
File	Purpose	Git-tracked?
`.env`	Your actual Gemini API key	❌ Excluded via `.gitignore`
`.env.example`	Placeholder template (no real secrets)	✅ Committed
The `.gitignore` also excludes:
`data/` — raw and processed dataset files
`models/` — trained model checkpoints
`outputs/` — experiment artefacts
`*.pt`, `*.pth`, `*.ckpt` — PyTorch checkpoint files
`Rare_Emotion_Detection_EDA.ipynb` — local EDA notebook
---
Research References
GoEmotions Dataset  
Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), 4040–4054.
NRC-VAD Lexicon  
Mohammad, S. M. (2018). Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words. Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018), 174–184.
RoBERTa  
Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv:1907.11692.
Multi-Task Learning for NLP  
Caruana, R. (1997). Multitask learning. Machine Learning, 28(1), 41–75.
Affective Computing and VAD Space  
Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161–1178.
Imbalanced Text Classification  
Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. Neural Networks, 106, 249–259.
---
Future Work
Complete Conditions B–D — Run oversampling, unfiltered-synthetic, and VAD-filtered-synthetic experiments to enable cross-condition comparison.
Per-class threshold calibration — Tune sigmoid thresholds independently per emotion on the validation split, particularly for rare classes.
Calibration analysis — Compute Expected Calibration Error (ECE) and Brier score for all conditions.
Larger synthetic set — Generate substantially more synthetic candidates (e.g., 100–500 per rare emotion) to give augmentation a meaningful signal.
Class-weighted BCE — Experiment with inverse-frequency class weights in the BCE loss as an alternative to synthetic augmentation.
GPU training — Re-run all conditions on a GPU to obtain realistic wall-clock estimates and complete multi-epoch training.
Few-shot prompting strategies — Investigate chain-of-thought or emotion-definition prompting to improve Gemini-generated candidate quality.
Cross-dataset generalisation — Evaluate the trained model on external emotion datasets (e.g., EmoBank, SemEval-2018 Task 1) to measure generalisation.
---
This repository is part of an ongoing capstone research project. All experimental results reported above are derived from completed runs only. Conditions B, C, and D and corresponding analyses will be updated in-place upon completion. Words," Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL), 2018.
- **RoBERTa**: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach," arXiv:1907.11692, 2019.
