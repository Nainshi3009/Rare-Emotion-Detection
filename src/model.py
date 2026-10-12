"""RoBERTa multi-label emotion classifier with a VAD pseudo-target regression head.

VAD targets are NRC-VAD V1 lexicon-derived pseudo-targets, not
human-annotated sentence-level VAD ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import RobertaModel


@dataclass
class JointModelOutput:
    emotion_logits: torch.Tensor
    vad_predictions: torch.Tensor
    loss: torch.Tensor | None = None
    classification_loss: torch.Tensor | None = None
    vad_loss: torch.Tensor | None = None


class RobertaEmotionVADModel(nn.Module):
    """RoBERTa encoder with 28 emotion logits and three VAD outputs.

    GoEmotions rows may have multiple labels. The classification objective is
    unweighted BCEWithLogitsLoss against original binary multi-hot targets.
    VAD values are lexicon-derived pseudo-targets, never human ground truth.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_labels: int = 28,
        lambda_vad: float = 1.0,
        dropout: float = 0.1,
        freeze_strategy: str = "full",
        *,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if num_labels != 28:
            raise ValueError("This project uses the original 28-class GoEmotions label space.")
        if lambda_vad < 0:
            raise ValueError("lambda_vad must be non-negative.")
        self.model_name = model_name
        self.num_labels = num_labels
        self.lambda_vad = float(lambda_vad)
        self.encoder = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
            local_files_only=local_files_only,
        )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden_size, num_labels)
        self.vad_head = nn.Linear(hidden_size, 3)
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        self.vad_loss_fn = nn.MSELoss(reduction="none")
        self.set_freeze_strategy(freeze_strategy)

    def set_freeze_strategy(self, strategy: str) -> None:
        """Set encoder/head trainability for the selected resource strategy."""
        valid_strategies = {"full", "frozen_encoder", "last_2_layers"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"freeze_strategy must be one of {sorted(valid_strategies)}; got {strategy!r}."
            )

        for parameter in self.parameters():
            parameter.requires_grad_(True)

        if strategy == "frozen_encoder":
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        elif strategy == "last_2_layers":
            layers = self.encoder.encoder.layer
            if len(layers) < 2:
                raise RuntimeError(
                    "The encoder must have at least two transformer layers "
                    "for freeze_strategy='last_2_layers'."
                )
            for parameter in self.encoder.embeddings.parameters():
                parameter.requires_grad_(False)
            for layer in layers[:-2]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

        self.freeze_strategy = strategy

    def parameter_report(self) -> dict[str, object]:
        """Summarize parameter counts and trainable module groups."""
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        groups: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                group = name.rsplit(".", 1)[0] if "." in name else name
                groups[group] = groups.get(group, 0) + parameter.numel()
        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": total_parameters - trainable_parameters,
            "percentage_trainable": (
                100.0 * trainable_parameters / total_parameters
                if total_parameters
                else 0.0
            ),
            "trainable_parameter_groups": groups,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emotion_targets: torch.Tensor | None = None,
        vad_targets: torch.Tensor | None = None,
        vad_valid_mask: torch.Tensor | None = None,
    ) -> JointModelOutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        shared = self.dropout(encoded.last_hidden_state[:, 0, :])
        emotion_logits = self.emotion_head(shared)
        vad_predictions = self.vad_head(shared)

        classification_loss = None
        if emotion_targets is not None:
            if emotion_targets.ndim != 2 or emotion_targets.shape[1] != self.num_labels:
                raise ValueError(
                    "emotion_targets must be binary multi-hot targets "
                    f"with shape [batch, {self.num_labels}]."
                )
            if not torch.all((emotion_targets == 0) | (emotion_targets == 1)):
                raise ValueError("emotion_targets must contain only 0/1 values.")
            classification_loss = self.classification_loss_fn(
                emotion_logits,
                emotion_targets.to(dtype=emotion_logits.dtype),
            )

        vad_loss = None
        if vad_targets is not None:
            if vad_targets.shape != vad_predictions.shape:
                raise ValueError("vad_targets must have shape [batch, 3].")
            if vad_valid_mask is None:
                raise ValueError("vad_valid_mask is required when VAD targets are provided.")
            valid = vad_valid_mask.to(dtype=torch.bool).reshape(-1)
            if valid.shape[0] != vad_predictions.shape[0]:
                raise ValueError("vad_valid_mask must have one value per batch row.")
            if valid.any():
                per_dimension_loss = self.vad_loss_fn(
                    vad_predictions[valid],
                    vad_targets[valid].to(dtype=vad_predictions.dtype),
                )
                vad_loss = per_dimension_loss.mean()
            else:
                vad_loss = vad_predictions.sum() * 0.0

        total_loss = None
        if classification_loss is not None:
            total_loss = classification_loss
            if vad_loss is not None:
                total_loss = total_loss + self.lambda_vad * vad_loss

        return JointModelOutput(
            emotion_logits=emotion_logits,
            vad_predictions=vad_predictions,
            loss=total_loss,
            classification_loss=classification_loss,
            vad_loss=vad_loss,
        )
