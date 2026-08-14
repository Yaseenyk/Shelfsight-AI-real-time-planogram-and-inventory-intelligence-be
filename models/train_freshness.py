"""Fine-tune the perishable freshness classifier (MobileNetV2 / ResNet50).

    python models/train_freshness.py --data-dir data/freshness --epochs 25
    python models/train_freshness.py --data-dir data/kaggle-fruits --dry-run

Dataset layouts (Kaggle / Roboflow / flat) and the folder→label mapping are
handled by `models/dataset.py`. Start with `--dry-run` to print exactly how each
source folder was mapped before spending an hour of GPU time on it.

Outputs
-------
- `models/weights/freshness_<backbone>.pt` — checkpoint carrying `state_dict`,
  `backbone`, `classes` and `input_size`, so inference never guesses label order.
- `evaluation/reports/freshness_val_predictions.json` — validation predictions in
  the shape `evaluation/benchmark.py freshness --labels` consumes.
- `evaluation/reports/freshness_training_history.json` — per-epoch curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services.freshness import IMAGENET_MEAN, IMAGENET_STD, build_backbone
from models.dataset import CANONICAL_CLASSES, DatasetError, FreshnessDataset, build_torch_dataset

logger = get_logger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the freshness classifier")
    parser.add_argument("--data-dir", required=True, help="dataset root (see models/dataset.py)")
    parser.add_argument(
        "--class-map", help="JSON overriding folder→label mapping for odd datasets"
    )
    parser.add_argument(
        "--backbone",
        default=settings.FRESHNESS_BACKBONE,
        choices=["mobilenet_v2", "resnet50"],
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=settings.FRESHNESS_INPUT_SIZE)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="train only the classifier head — right for small datasets (<2k images)",
    )
    parser.add_argument("--out", help="checkpoint path (default: models/weights/<backbone>.pt)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the dataset and print the class mapping, then exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)

    # ---- dataset resolution happens before torch, so --dry-run is instant ----
    try:
        dataset = FreshnessDataset.discover(
            Path(args.data_dir),
            class_map_file=Path(args.class_map) if args.class_map else None,
            val_split=args.val_split,
            classes=CANONICAL_CLASSES,
            seed=args.seed,
        )
    except DatasetError as exc:
        logger.error("%s", exc)
        return 1

    print(dataset.describe())
    if args.dry_run:
        return 0
    if not dataset.val:
        logger.error("No validation images — cannot measure accuracy. Adjust --val-split.")
        return 1

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import transforms
    except ImportError:
        logger.error("torch/torchvision missing — pip install -r requirements-ml.txt")
        return 1

    torch.manual_seed(args.seed)
    device = torch.device(
        settings.DETECTION_DEVICE if torch.cuda.is_available() else "cpu"
    )
    size = args.input_size
    logger.info("Training on %s", device)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Colour jitter stays mild on purpose: spoilage *is* a colour cue,
            # so aggressive hue/saturation augmentation trains the signal away.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = build_torch_dataset(dataset.train, train_tf)
    val_ds = build_torch_dataset(dataset.val, eval_tf)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers
    )
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    classes = dataset.classes
    model = build_backbone(
        args.backbone, len(classes), pretrained=settings.FRESHNESS_PRETRAINED
    ).to(device)

    if args.freeze_backbone:
        head = _classifier_parameters(model, args.backbone)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head:
            parameter.requires_grad = True
        logger.info("Backbone frozen — training the classifier head only")

    # Class weights counter the imbalance every public freshness set has
    # (rotten images are far rarer than fresh ones).
    weights = _class_weights(dataset.distribution("train"), classes)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = Path(args.out or settings.FRESHNESS_WEIGHTS)
    if not args.out:
        out_path = out_path.with_name(f"freshness_{args.backbone}.pt")

    best_accuracy = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()

        accuracy, predictions = _evaluate(model, val_dl, device, classes)
        train_loss = running / seen if seen else 0.0
        history.append(
            {"epoch": epoch, "train_loss": round(train_loss, 4), "val_accuracy": round(accuracy, 4)}
        )
        logger.info("epoch %02d | loss %.4f | val acc %.4f", epoch, train_loss, accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "classes": classes,
                    "input_size": size,
                    "val_accuracy": accuracy,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "dataset": str(dataset.root),
                    "seed": args.seed,
                },
                out_path,
            )
            settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            (settings.REPORTS_DIR / "freshness_val_predictions.json").write_text(
                json.dumps({"samples": predictions}, indent=2), encoding="utf-8"
            )
            logger.info("  ↳ new best, checkpoint saved to %s", out_path)

    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (settings.REPORTS_DIR / "freshness_training_history.json").write_text(
        json.dumps(
            {
                "backbone": args.backbone,
                "classes": classes,
                "dataset": str(dataset.root),
                "distribution": dataset.distribution("train"),
                "best_val_accuracy": best_accuracy,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Best val accuracy %.4f → %s", best_accuracy, out_path)
    logger.info(
        "Score it: python -m evaluation.benchmark freshness "
        "--labels evaluation/reports/freshness_val_predictions.json"
    )
    return 0


def _evaluate(model: Any, loader: Any, device: Any, classes: List[str]):  # noqa: ANN202
    """Validation pass returning `(accuracy, benchmark-shaped predictions)`."""
    import torch  # noqa: PLC0415

    model.eval()
    correct = total = 0
    predictions: List[Dict[str, str]] = []
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predicted == labels).sum())
            total += labels.size(0)
            predictions.extend(
                {"truth_label": classes[t], "predicted_label": classes[p]}
                for t, p in zip(labels.tolist(), predicted.tolist())
            )
    return (correct / total if total else 0.0), predictions


def _class_weights(distribution: Dict[str, int], classes: List[str]) -> List[float]:
    """Inverse-frequency weights; absent classes get 0 so they cannot skew loss."""
    counts = [max(0, distribution.get(name, 0)) for name in classes]
    present = [c for c in counts if c > 0]
    if not present:
        return [1.0] * len(classes)
    mean_count = sum(present) / len(present)
    return [(mean_count / count) if count > 0 else 0.0 for count in counts]


def _classifier_parameters(model: Any, backbone: str) -> List[Any]:
    if backbone == "resnet50":
        return list(model.fc.parameters())
    return list(model.classifier.parameters())


if __name__ == "__main__":
    sys.exit(main())
