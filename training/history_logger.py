"""
Before the epoch loop:
    history = TrainingHistory()

At the end of every epoch:
    history.add(
        epoch=epoch,
        train_loss=train_loss,
        validation_loss=validation_loss,
    )
    history.save(CFG.results_dir / f"{model_name}_history.csv")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class TrainingHistory:
    rows: list[dict[str, float | int]] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
    ) -> None:
        self.rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "perplexity": float(
                    math.exp(min(float(validation_loss), 20.0))
                ),
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(path, index=False)
