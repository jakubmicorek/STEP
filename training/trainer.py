"""training/trainer.py - STEPTrainer: EBM training loop with EMA and LR scheduling."""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .loss import dsm_loss


def ema_update(model: torch.nn.Module, model_ema: torch.nn.Module, decay: float):
    with torch.no_grad():
        for (name, p), p_ema in zip(model.named_parameters(), model_ema.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class STEPTrainer:
    """Training loop: runs epochs, updates EMA, fires val/save callbacks with the EMA model."""

    def __init__(self, model, model_ema, optimizer, scheduler, args, writer=None):
        self.model     = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args      = args
        self.writer    = writer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, features: np.ndarray, confs: np.ndarray, val_callback=None, periodic_save_fn=None):
        """Train for args.epochs epochs; fires callbacks with the EMA model."""
        ds = TensorDataset(
            torch.from_numpy(features).float(),
            torch.from_numpy(confs).float(),
        )
        loader = DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        steps_per_epoch = len(loader)
        global_step     = 0
        args            = self.args

        for epoch in range(args.epochs):
            self.model.train()
            pbar       = tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{args.epochs}")
            epoch_loss = 0.0

            for batch_idx, (x, batch_confs) in enumerate(pbar):
                x           = x.to(args.device)
                batch_confs = batch_confs.to(args.device)

                # Linear LR warmup over the first epoch
                if epoch == 0:
                    frac = (batch_idx + 1) / steps_per_epoch
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = args.lr * frac

                loss = dsm_loss(
                    self.model,
                    x,
                    batch_confs,
                    confidence_weighted=args.train_confidence_weighted,
                )

                self.optimizer.zero_grad()
                loss.backward()

                if args.gradient_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.gradient_clipping
                    )
                else:
                    grad_norm = torch.stack([
                        p.grad.norm() for p in self.model.parameters() if p.grad is not None
                    ]).norm()

                self.optimizer.step()

                if args.ema_decay > 0:
                    ema_update(self.model, self.model_ema, args.ema_decay)

                loss_val    = loss.item()
                epoch_loss += loss_val
                global_step += 1

                if self.writer and global_step % 50 == 0:
                    self.writer.add_scalar("Train/Loss_step",  loss_val,                        global_step)
                    self.writer.add_scalar("Train/LR",         self.optimizer.param_groups[0]["lr"], global_step)
                    self.writer.add_scalar("Train/GradNorm",   grad_norm.item(),                global_step)

                pbar.set_postfix(loss=f"{loss_val:.4f}")

            if self.scheduler and epoch > 0:
                self.scheduler.step()

            if self.writer:
                self.writer.add_scalar("Train/Loss_epoch", epoch_loss / steps_per_epoch, epoch + 1)
                self.writer.add_scalar("Train/LR_epoch",   self.optimizer.param_groups[0]["lr"], epoch + 1)

            if (
                val_callback is not None
                and args.check_val_every_n_epoch > 0
                and (epoch + 1) % args.check_val_every_n_epoch == 0
            ):
                inference_model = self.model_ema if args.ema_decay > 0 else self.model
                val_callback(inference_model, epoch + 1)
                self.model.train()

            if (
                periodic_save_fn is not None
                and getattr(args, "save_every_n_epochs", 0) > 0
                and (epoch + 1) % args.save_every_n_epochs == 0
            ):
                inference_model = self.model_ema if args.ema_decay > 0 else self.model
                periodic_save_fn(inference_model, epoch + 1)

        return self.model_ema if self.args.ema_decay > 0 else self.model

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def build(model, args, writer=None):
        """Construct trainer with AdamW + optional cosine annealing."""
        model_ema = copy.deepcopy(model)
        model_ema.eval()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.9),
        )

        scheduler = None
        if args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.lr / 2,
            )

        return STEPTrainer(model, model_ema, optimizer, scheduler, args, writer=writer)
