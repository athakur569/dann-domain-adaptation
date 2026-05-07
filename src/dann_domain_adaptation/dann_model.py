"""DANN model: task classifier + domain discriminator with GRL."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models

from .grl import GradientReversalLayer


class DANNModel(pl.LightningModule):
    """Domain-Adversarial Neural Network.

    Architecture:
        Encoder (ResNet-50 backbone, no final FC)
        ├── Task classifier:   FC → n_classes
        └── Domain classifier: GRL → FC → FC → n_domains

    Args:
        n_classes:   Number of task classes.
        n_domains:   Number of domain labels (e.g. 5 for CAMELYON17 centers).
        pretrained:  Load ImageNet weights.
        lr:          Learning rate.
        gamma:       Lambda schedule steepness.
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_domains: int = 5,
        pretrained: bool = True,
        lr: float = 1e-4,
        gamma: float = 10.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # strip FC
        feat_dim = 2048

        self.task_head = nn.Linear(feat_dim, n_classes)
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, n_domains),
        )

        self.task_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, lambda_: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.grl.lambda_ = lambda_
        feats = self.encoder(x).flatten(1)
        task_logits = self.task_head(feats)
        domain_logits = self.domain_head(self.grl(feats))
        return task_logits, domain_logits

    def _lambda(self) -> float:
        p = self.current_epoch / max(self.trainer.max_epochs, 1)
        return GradientReversalLayer.lambda_schedule(p, self.hparams.gamma)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y_task, y_domain = batch
        lam = self._lambda()
        task_logits, domain_logits = self(x, lambda_=lam)
        loss = self.task_loss(task_logits, y_task) + self.domain_loss(domain_logits, y_domain)
        self.log("train/loss", loss)
        self.log("train/lambda", lam)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
