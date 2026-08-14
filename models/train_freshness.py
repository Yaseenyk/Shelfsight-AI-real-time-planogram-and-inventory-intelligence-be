"""Train the perishable freshness classifier (MobileNetV2 / ResNet50).

    python models/train_freshness.py --data-dir data/freshness --epochs 25

Expects an ImageFolder layout:

    data/freshness/train/{fresh,ripening,spoiled}/*.jpg
    data/freshness/val/{fresh,ripening,spoiled}/*.jpg

The checkpoint stores `state_dict`, `backbone` and `classes` so inference never
has to guess the label order. Validation predictions are dumped in the exact
shape `evaluation/benchmark.py freshness --labels` consumes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services.freshness import IMAGENET_MEAN, IMAGENET_STD, build_backbone

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the freshness classifier")
    parser.add_argument("--data-dir", required=True, help="ImageFolder root with train/ and val/")
    parser.add_argument("--backbone", default=settings.FRESHNESS_BACKBONE,
                        choices=["mobilenet_v2", "resnet50"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)  # 0 is safest on Windows
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(settings.FRESHNESS_WEIGHTS))
    return parser.parse_args()


def main() -> int:
    configure_logging()
    args = parse_args()

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError:
        logger.error("torch/torchvision missing — pip install -r requirements-ml.txt")
        return 1

    torch.manual_seed(args.seed)
    device = torch.device(settings.DETECTION_DEVICE if torch.cuda.is_available() else "cpu")
    size = settings.FRESHNESS_INPUT_SIZE

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            # Colour jitter matters here: spoilage IS a colour cue, so keep it mild.
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

    root = Path(args.data_dir)
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    classes = train_ds.classes
    logger.info("Classes: %s | train=%d val=%d", classes, len(train_ds), len(val_ds))

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_backbone(args.backbone, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_accuracy = 0.0
    out_path = Path(args.out)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        scheduler.step()

        model.eval()
        correct = total = 0
        predictions = []
        with torch.no_grad():
            for images, labels in val_dl:
                images = images.to(device)
                predicted = model(images).argmax(dim=1).cpu()
                correct += int((predicted == labels).sum())
                total += labels.size(0)
                predictions.extend(
                    {"truth_label": classes[t], "predicted_label": classes[p]}
                    for t, p in zip(labels.tolist(), predicted.tolist())
                )

        accuracy = correct / total if total else 0.0
        train_loss = running / len(train_ds) if len(train_ds) else 0.0
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_accuracy": round(accuracy, 4)})
        logger.info("epoch %02d | loss %.4f | val acc %.4f", epoch, train_loss, accuracy)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "classes": classes,
                    "val_accuracy": accuracy,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                },
                out_path,
            )
            settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            (settings.REPORTS_DIR / "freshness_val_predictions.json").write_text(
                json.dumps({"samples": predictions}, indent=2), encoding="utf-8"
            )

    logger.info("Best val accuracy %.4f -> %s", best_accuracy, out_path)
    (settings.REPORTS_DIR / "freshness_training_history.json").write_text(
        json.dumps({"backbone": args.backbone, "history": history}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
