"""Utility to evaluate a trained classifier checkpoint on the DDI dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC

from .constants import build_cfg_from_cli
from .datamodule import FitzpatrickDataModule
from .lightning_module import FitzpatrickClassifier


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Fitzpatrick classifier checkpoint on the DDI dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the Lightning checkpoint produced by train_classifier.py.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="Optional YAML config file with overrides.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help=(
            "Path for the CSV containing DDI identifiers and predicted labels. "
            "Defaults to <PATHS.ROOT>/ddi_predictions.csv."
        ),
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Optional KEY=VALUE pairs to override config options.",
    )
    return parser.parse_args()


def _build_cfg(args) -> object:
    cfg_args: list[str] = []
    if args.config_file:
        cfg_args.extend(["--config-file", args.config_file])
    if args.opts:
        cfg_args.extend(args.opts)
    return build_cfg_from_cli(cfg_args)


def _resolve_device(device_str: str) -> torch.device:
    target = (device_str or "cpu").lower()
    if target in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[evaluate_ddi] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if target == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[evaluate_ddi] MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def evaluate_checkpoint_on_ddi(cfg, checkpoint_path: Path):
    checkpoint_path = checkpoint_path.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    dm = FitzpatrickDataModule(cfg)
    dm.setup("test")
    dataloader = dm.test_dataloader()
    if dataloader is None or dm.test_dataset is None:
        raise RuntimeError("Failed to build the DDI test dataloader.")
    ddi_dataset = dm.test_dataset

    model = FitzpatrickClassifier.load_from_checkpoint(str(checkpoint_path), cfg=cfg)
    device = _resolve_device(cfg.INFERENCE.DEVICE)
    model.to(device)
    model.eval()

    accuracy = MulticlassAccuracy(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    f1_score = MulticlassF1Score(num_classes=cfg.MODEL.NUM_CLASSES).to(device)
    auroc = MulticlassAUROC(num_classes=cfg.MODEL.NUM_CLASSES).to(device)

    total_loss = 0.0
    total_examples = 0
    predictions: list[dict[str, str]] = []
    running_index = 0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device, dtype=model.weight_dtype)
            labels = labels.to(device)
            logits = model(images)
            loss = model.criterion(logits, labels)

            batch_size = labels.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            accuracy.update(preds, labels)
            f1_score.update(preds, labels)
            auroc.update(probs, labels)

            batch_ids = ddi_dataset.df.iloc[
                running_index : running_index + batch_size
            ][ddi_dataset.img_col].tolist()
            running_index += batch_size
            pred_indices = preds.cpu().tolist()
            label_indices = labels.cpu().tolist()
            for identifier, pred_idx, label_idx in zip(
                batch_ids, pred_indices, label_indices
            ):
                predictions.append(
                    {
                        "ddi_identifier": identifier,
                        "generated_label": ddi_dataset.labels[pred_idx],
                        "true_label": ddi_dataset.labels[label_idx],
                    }
                )

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy.compute().item(),
        "f1": f1_score.compute().item(),
        "auc": auroc.compute().item(),
        "num_examples": total_examples,
    }
    return metrics, predictions


def _write_predictions_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ddi_identifier", "generated_label", "true_label"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = _parse_args()
    cfg = _build_cfg(args)
    metrics, predictions = evaluate_checkpoint_on_ddi(cfg, Path(args.checkpoint))

    csv_path = (
        Path(args.output_csv).expanduser()
        if args.output_csv
        else Path(cfg.PATHS.ROOT) / "ddi_predictions.csv"
    )
    _write_predictions_csv(csv_path, predictions)

    print(
        f"[evaluate_ddi] Evaluated {metrics['num_examples']} DDI images "
        f"using checkpoint '{args.checkpoint}'."
    )
    print(
        f"[evaluate_ddi] Loss: {metrics['loss']:.4f} | "
        f"AUC: {metrics['auc']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )
    print(f"[evaluate_ddi] Saved DDI predictions to '{csv_path}'.")


if __name__ == "__main__":
    main()
