# GoEmotions multi-label loss methodology audit

## Scope

This audit describes the current data target construction, joint model loss, and
evaluation implementation. No loss, model, configuration, or experimental
condition was changed as part of this audit. GoEmotions has 28 emotion labels;
approximately 7,102 of 43,410 training rows contain more than one label.

## Current implementation

Each source row stores its labels as a list of integer IDs in the original
0–27 label space. In `src/dataset.py`, those IDs are expanded to a 28-element
multi-hot vector, `emotion_labels`:

- `1` for every annotated label
- `0` for every other label

The same vector is divided by the number of positive labels to produce
`emotion_targets`. A row with labels `{a, b}` therefore has target values
`0.5` at `a` and `b`, and zero elsewhere. A single-label row has one target
value of `1.0`. Both `emotion_labels` and `emotion_targets` have shape
`[batch_size, 28]`.

The model emits raw, unnormalized `emotion_logits` of shape
`[batch_size, 28]`. During training, `nn.CrossEntropyLoss` is applied to those
logits and the floating-point, normalized multi-hot `emotion_targets`.
PyTorch's cross-entropy applies `log_softmax` internally; the model does not
apply sigmoid or softmax before this loss. The joint objective is:

`total_loss = classification_loss + lambda_vad * vad_loss`

where VAD regression uses masked mean squared error against NRC-VAD V1
lexicon-derived pseudo-targets.

For evaluation, `training.py` applies sigmoid independently to each logit and
predicts every label whose probability is at least the configured threshold
(currently 0.5). It compares those predictions to the original, unnormalized
multi-hot targets. Per-class F1 is calculated across the 28 label indicators;
macro-F1 is the unweighted mean of the 28 per-class F1 values, and weighted-F1
weights class F1 by its support. Rare-target F1 selects the per-class values
for grief, pride, relief, and nervousness. These metrics do not use softmax or
force exactly one predicted class.

One reporting note in the current metric metadata calls these “micro-example
metrics.” The implementation actually computes label-wise macro and weighted
F1 from a multi-label indicator matrix; it does not request scikit-learn's
`average="samples"` or `average="micro"`. This wording should be corrected
separately when metric documentation is next updated.

## Mathematical assessment

### Cross-entropy with normalized multi-hot targets

For logits `z` and normalized target distribution `q`, the current
classification loss for one row is:

`L_CE = -sum_i q_i * log(softmax(z)_i)`

If there are `k` positive labels, each has target mass `1/k`. This is a valid
soft-target cross-entropy objective: targets are finite, nonnegative, and sum
to one, satisfying the probability-target interpretation expected by
`CrossEntropyLoss`.

It is **not the standard independent-label multi-label objective**, however.
Softmax probabilities sum to one across all 28 classes, so labels compete for
a fixed probability mass. For a multi-label row, the loss encourages the
positive labels to share that mass. It does not independently ask the model
to mark each positive label present and each negative label absent. Meanwhile,
evaluation uses independent sigmoid probabilities and a fixed threshold.
Thus the current training objective and prediction rule have different
probabilistic semantics.

### BCEWithLogitsLoss with multi-hot targets

The usual formulation for independent binary labels applies
`BCEWithLogitsLoss` directly to raw logits and the **unnormalized**
multi-hot `emotion_labels`:

`L_BCE = mean_i[-y_i*log(sigmoid(z_i)) - (1-y_i)*log(1-sigmoid(z_i))]`

Each label is modeled as a separate binary decision. This directly aligns the
training loss with sigmoid-plus-threshold inference and standard multi-label
F1 evaluation. The existing dataset already retains the required
unnormalized multi-hot tensor.

Because each example has 28 label decisions and usually far fewer positives,
plain mean BCE can be dominated by negatives. If BCE is adopted, the handling
of class imbalance (for example, plain BCE versus `pos_weight`, focal loss, or
another explicitly justified strategy) must be decided and held consistently
across Conditions A–D. Introducing class weights would change the objective
and requires separate methodological justification; it should not be slipped
into the loss conversion.

## Comparison for this project

| Formulation | Target | Output interpretation | Advantages | Limitations |
|---|---|---|---|---|
| Current soft-target cross-entropy | Normalized multi-hot, each row sums to 1 | Mutually exclusive categorical distribution | Mathematically valid probability-target CE; gives equal target mass to every annotation; keeps every positive label in the target | Imposes competition among co-occurring emotions; inconsistent with independent sigmoid thresholding; not the conventional independent-label multi-label loss |
| BCE with logits | Unnormalized multi-hot 0/1 vector | Independent label probabilities | Standard multi-label formulation; directly matches sigmoid thresholding and label-wise F1; existing `emotion_labels` provides the target | Many negatives per row can dominate plain BCE; any weighting or rebalancing changes the objective and needs a documented choice |

The current CE is mathematically valid for the *normalized target
distribution*, but that distribution is an artificial normalization of
co-occurring annotations. It is therefore a defensible alternative objective
only if the intended task is reframed as distributing probability mass across
annotated labels. Given the existing GoEmotions multi-label representation and
the project's independent sigmoid/F1 evaluation, it is not the most
methodologically aligned choice.

## Recommendation and impact

For a multi-label rare-emotion detection study, use
`BCEWithLogitsLoss` with the original unnormalized 28-dimensional
multi-hot targets, with an explicit decision about whether the unweighted
baseline is sufficient. Do not divide positive labels by the number of labels
for BCE. Keep the same classification objective and any weighting policy in
all experimental conditions A–D.

Changing from CE to BCE changes the training objective and therefore the
methodology; it is not a mechanical implementation detail. It should happen
before any real condition is trained, be documented in the experiment
configuration/report, and be applied uniformly to every condition. Existing
sanity-check and freezing resource benchmark values do not establish model
quality under either formulation and should not be compared as scientific
results.

The current sigmoid thresholding and multi-hot F1 evaluation are structurally
compatible with BCE and do not require a change merely to support it.
Threshold selection/calibration is a separate methodological decision: if
the fixed 0.5 threshold is retained, state that explicitly; if thresholds
are tuned, tune them only on validation data and use the same protocol for
every condition. Also correct the “micro-example” metric note to describe
label-wise macro/weighted F1 accurately.

## Selected Experimental Formulation

Following the audit, the selected formulation for upcoming experiments is:

- Original 28-dimensional, unnormalized binary multi-hot targets (values only
  `0` or `1`).
- Unweighted `BCEWithLogitsLoss` applied directly to raw emotion logits; no
  sigmoid is applied before the loss.
- Sigmoid-based evaluation at the fixed threshold `0.5`, with per-label
  precision, recall, and F1; macro-F1, weighted-F1, and rare-emotion F1 for
  grief, pride, relief, and nervousness.
- Masked MSE for VAD regression, using `vad_valence`, `vad_arousal`, and
  `vad_dominance` only where the VAD-valid mask is true.
- `total_loss = BCEWithLogitsLoss + lambda_vad * masked_VAD_MSE`, with
  `lambda_vad = 0.5`.
- No class weighting in the baseline. Rare-class performance will be measured
  using per-class F1 and rare-emotion F1. Any later class weighting is a
  separate methodological experiment and must be documented and applied
  consistently when compared.

This replaces the previously audited normalized-target
`CrossEntropyLoss` formulation before any real experiment. Evaluation remains
independent-label sigmoid evaluation and does not use softmax. No real model
training was performed as part of this methodology change.
