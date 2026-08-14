"""Torch `Dataset` for freshness training.

Separate module, imported only when training, because it imports torch at module
level — `models/dataset.py` must stay importable without the ML stack (the
curator and most tests depend on it).

**The class lives at module scope on purpose.** Defining it inside a factory
function makes it unpicklable, and `DataLoader(num_workers>0)` on Windows uses
the *spawn* start method, which pickles the dataset to hand it to each worker.
That fails with `Can't pickle local object`, which reads like a multiprocessing
bug and is really a scoping one. Module scope keeps multi-worker loading — the
difference between a 10-minute epoch and an hour — available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch
from PIL import Image as PILImage

from app.core.logging import get_logger
from app.utils.vision import ImageDecodeError, read_image_file, to_rgb

logger = get_logger(__name__)


class FreshnessDataset(torch.utils.data.Dataset):
    """`(path, label)` pairs read through the same ingestion path as the API.

    Using `app.utils.vision.read_image_file` means training never sees an image
    the API would reject, and a corrupt file is skipped rather than killing an
    epoch twenty minutes in.
    """

    def __init__(self, items: Sequence[Tuple[Path, int]], transform: Any) -> None:
        self.items: List[Tuple[Path, int]] = [(Path(p), int(label)) for p, label in items]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """Return the sample, skipping forward past unreadable files.

        Iterative, not recursive, and bounded by the dataset length. A recursive
        version blows the stack when *many* files are unreadable — e.g. a wrong
        `--data-dir`, where every path fails — turning a clear "no readable
        images" error into an opaque RecursionError.
        """
        total = len(self.items)
        first_error: Any = None

        for offset in range(total):
            position = (index + offset) % total
            path, label = self.items[position]
            try:
                array = to_rgb(read_image_file(path))
            except ImageDecodeError as exc:
                if first_error is None:
                    first_error = exc
                logger.warning("Skipping unreadable image %s: %s", path, exc)
                continue
            return self.transform(PILImage.fromarray(array)), label

        raise RuntimeError(
            f"No readable images in the dataset ({total} tried). "
            f"First failure: {first_error}"
        )
