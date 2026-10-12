from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

import src.training as training


class _TinyDataset(Dataset):
    def __init__(self, frame, tokenizer, *, max_length: int) -> None:
        self.size = len(frame)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([index + 1], dtype=torch.float32),
            "attention_mask": torch.ones(1, dtype=torch.long),
            "emotion_targets": torch.zeros(28),
            "emotion_labels": torch.zeros(28, dtype=torch.long),
            "vad_targets": torch.zeros(3),
            "vad_valid_mask": torch.tensor(False),
        }


class _TinyModel(nn.Module):
    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        super().__init__()
        self.model_name = "tiny-test-model"
        self.num_labels = 28
        self.lambda_vad = 0.5
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.calls = 0
        self.seen_inputs: list[int] = []
        self.interrupt_on_call = interrupt_on_call

    def set_freeze_strategy(self, strategy: str) -> None:
        if strategy != "full":
            raise AssertionError(f"Unexpected test freeze strategy: {strategy}")

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        if self.calls == self.interrupt_on_call:
            raise KeyboardInterrupt
        self.seen_inputs.append(int(input_ids[0, 0].item()))
        loss = (self.weight * input_ids.float().mean() - 1.0).square()
        return SimpleNamespace(
            loss=loss,
            classification_loss=loss,
            vad_loss=loss * 0.0,
        )


class TrainingResumeSmokeTest(unittest.TestCase):
    def test_interrupted_run_resumes_at_saved_microbatch_and_global_step(self) -> None:
        train_frame = pd.DataFrame(
            {
                "text": [f"training row {index}" for index in range(5)],
                "labels": [[0] for _ in range(5)],
            }
        )
        validation_frame = pd.DataFrame(
            {"text": ["validation row"], "labels": [[0]]}
        )
        config = training.TrainingConfig(
            model_name="tiny-test-model",
            freeze_strategy="full",
            max_length=8,
            batch_size=1,
            epochs=1,
            learning_rate=0.01,
            warmup_ratio=0.0,
            gradient_accumulation_steps=2,
            cpu_num_threads=1,
        )
        expected_order = [
            int(batch["input_ids"][0, 0].item())
            for batch in DataLoader(
                _TinyDataset(train_frame, None, max_length=config.max_length),
                batch_size=config.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(config.seed),
            )
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recovery_path = root / "recovery" / "latest.pt"
            best_path = root / "best_model.pt"
            patches = (
                patch.object(training, "GoEmotionsVADDataset", _TinyDataset),
                patch.object(
                    training,
                    "evaluate_model",
                    return_value={
                        "macro_f1": 0.5,
                        "weighted_f1": 0.5,
                        "rare_target_f1": {},
                    },
                ),
            )
            with patches[0], patches[1]:
                interrupted_model = _TinyModel(interrupt_on_call=3)
                with self.assertRaises(KeyboardInterrupt):
                    training.train_model(
                        interrupted_model,
                        None,
                        train_frame,
                        validation_frame,
                        config=config,
                        checkpoint_path=best_path,
                        recovery_checkpoint_path=recovery_path,
                        recovery_interval_steps=1,
                    )

                saved = torch.load(
                    recovery_path,
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(saved["next_microbatch_position"], 2)
                self.assertEqual(saved["global_optimizer_step"], 1)
                self.assertEqual(saved["epoch_index"], 0)
                self.assertIn("optimizer_state_dict", saved)
                self.assertIn("scheduler_state_dict", saved)
                self.assertIn("rng_state", saved)

                resumed_model = _TinyModel()
                result = training.train_model(
                    resumed_model,
                    None,
                    train_frame,
                    validation_frame,
                    config=config,
                    checkpoint_path=best_path,
                    recovery_checkpoint_path=recovery_path,
                    resume_from_recovery=True,
                    recovery_interval_steps=1,
                )

            self.assertTrue(result["resumed_from_recovery"])
            self.assertEqual(result["global_optimizer_step"], 3)
            self.assertEqual(result["history"][0]["optimizer_steps"], 3)
            self.assertEqual(resumed_model.calls, 3)
            self.assertEqual(
                interrupted_model.seen_inputs + resumed_model.seen_inputs,
                expected_order,
            )
            self.assertTrue(best_path.is_file())
            self.assertTrue(recovery_path.is_file())

    def test_checkpoint_restores_optimizer_scheduler_rng_and_accumulated_gradients(self) -> None:
        config = training.TrainingConfig(
            model_name="tiny-test-model",
            freeze_strategy="full",
            epochs=1,
            learning_rate=0.01,
            warmup_ratio=0.0,
            gradient_accumulation_steps=2,
        )
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        model.weight.grad = torch.tensor(0.75)
        optimizer.step()
        model.weight.grad = torch.tensor(0.375)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=4,
        )
        saved_model_weight = model.weight.detach().clone()
        saved_optimizer_state = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[model.weight].items()
        }
        saved_scheduler_state = scheduler.state_dict().copy()
        expected_gradient = model.weight.grad.clone()
        generator = torch.Generator().manual_seed(71)
        loader_generator_state = generator.get_state()
        epoch_generator_state = torch.Generator().manual_seed(29).get_state()

        random.seed(1001)
        np.random.seed(1002)
        torch.manual_seed(1003)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "recovery.pt"
            training._save_recovery_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                config=config,
                train_fingerprint="train-hash",
                validation_fingerprint="validation-hash",
                epoch_index=0,
                next_microbatch_position=3,
                global_optimizer_step=1,
                best_macro_f1=0.25,
                best_checkpoint=None,
                epoch_loader_generator_state=epoch_generator_state,
                epoch_loss=2.0,
                epoch_classification_loss=1.5,
                epoch_vad_loss=1.0,
                epoch_microbatches=3,
                accumulation_position=1,
                history=[],
                generator=generator,
            )

            expected_rng = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )
            with torch.no_grad():
                model.weight.add_(10.0)
            model.weight.grad = torch.tensor(-3.0)
            optimizer.state[model.weight]["exp_avg"].add_(10.0)
            optimizer.step()
            scheduler.last_epoch += 4
            scheduler._step_count += 4
            random.seed(5)
            np.random.seed(6)
            torch.manual_seed(7)

            restored = training._load_recovery_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                config=config,
                train_fingerprint="train-hash",
                validation_fingerprint="validation-hash",
                device=torch.device("cpu"),
            )

            self.assertTrue(torch.equal(model.weight.detach(), saved_model_weight))
            self.assertTrue(torch.equal(model.weight.grad, expected_gradient))
            for key, value in saved_optimizer_state.items():
                actual = optimizer.state[model.weight][key]
                if torch.is_tensor(value):
                    self.assertTrue(torch.equal(actual, value), key)
                else:
                    self.assertEqual(actual, value, key)
            self.assertEqual(scheduler.state_dict(), saved_scheduler_state)
            self.assertTrue(
                torch.equal(
                    restored["epoch_loader_generator_state"],
                    epoch_generator_state,
                )
            )
            self.assertTrue(
                torch.equal(restored["loader_generator_state"], loader_generator_state)
            )
            self.assertEqual(
                (
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(())),
                ),
                expected_rng,
            )
            self.assertEqual(restored["next_microbatch_position"], 3)
            self.assertEqual(restored["global_optimizer_step"], 1)
            self.assertEqual(restored["accumulation_position"], 1)

    def test_recovery_checkpoint_rejects_config_and_data_mismatches(self) -> None:
        config = training.TrainingConfig(
            model_name="tiny-test-model",
            freeze_strategy="full",
            epochs=1,
            learning_rate=0.01,
            warmup_ratio=0.0,
            gradient_accumulation_steps=2,
        )
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=2,
        )
        generator = torch.Generator().manual_seed(9)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "recovery.pt"
            training._save_recovery_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                config=config,
                train_fingerprint="train-hash",
                validation_fingerprint="validation-hash",
                epoch_index=0,
                next_microbatch_position=0,
                global_optimizer_step=0,
                best_macro_f1=float("-inf"),
                best_checkpoint=None,
                epoch_loader_generator_state=generator.get_state(),
                epoch_loss=0.0,
                epoch_classification_loss=0.0,
                epoch_vad_loss=0.0,
                epoch_microbatches=0,
                accumulation_position=0,
                history=[],
                generator=generator,
            )

            mismatched_config = training.TrainingConfig(
                **{**training.asdict(config), "learning_rate": 0.02}
            )
            for requested_config, train_hash, validation_hash, message in (
                (mismatched_config, "train-hash", "validation-hash", "configuration"),
                (config, "changed-train-hash", "validation-hash", "training data"),
                (config, "train-hash", "changed-validation-hash", "validation data"),
            ):
                with self.subTest(mismatch=message):
                    with self.assertRaisesRegex(ValueError, message):
                        training._load_recovery_checkpoint(
                            checkpoint_path,
                            model,
                            optimizer,
                            scheduler,
                            config=requested_config,
                            train_fingerprint=train_hash,
                            validation_fingerprint=validation_hash,
                            device=torch.device("cpu"),
                        )

    def test_torch_rng_restore_passes_cpu_byte_tensor(self) -> None:
        state = training._capture_rng_state()
        captured: list[torch.Tensor] = []

        def assert_cpu_state(rng_state: torch.Tensor) -> None:
            captured.append(rng_state)
            self.assertEqual(rng_state.device.type, "cpu")
            self.assertEqual(rng_state.dtype, torch.uint8)

        with patch.object(training.torch, "set_rng_state", side_effect=assert_cpu_state):
            training._restore_rng_state(state)

        self.assertEqual(len(captured), 1)

    def test_cuda_rng_restore_passes_cpu_byte_tensors(self) -> None:
        cuda_state = torch.arange(8, dtype=torch.uint8)
        captured: list[list[torch.Tensor]] = []

        def assert_cpu_states(rng_states: list[torch.Tensor]) -> None:
            captured.append(rng_states)
            self.assertTrue(
                all(state.device.type == "cpu" for state in rng_states)
            )
            self.assertTrue(
                all(state.dtype == torch.uint8 for state in rng_states)
            )

        with (
            patch.object(training.torch.cuda, "is_available", return_value=True),
            patch.object(
                training.torch.cuda,
                "set_rng_state_all",
                side_effect=assert_cpu_states,
            ),
        ):
            training._restore_rng_state(
                {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": [cuda_state],
                }
            )

        self.assertEqual(len(captured), 1)


if __name__ == "__main__":
    unittest.main()
