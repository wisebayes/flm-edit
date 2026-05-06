"""Lightning module base for text editing with FLM/FMLM.

Adapted from trainer_base.py in https://github.com/david3684/flm.
Key differences:
- training_step unpacks (source_ids, target_ids, instruction_ids) dicts
- backbone is DITEdit instead of DIT
- forward() threads x_src_ids through to the backbone
- validation logs source→edit text pairs as a W&B table
"""
import itertools
import os
import random
import inspect

from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.utils.data import DataLoader
from omegaconf import ListConfig

import metrics
import models
import utils
from models.dit_edit import DITEdit
from trainer_base import Loss, LogLinear, sample_categorical, _unsqueeze


class TrainerEditBase(L.LightningModule):
    """Base Lightning module for edit-conditioned FLM variants."""

    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer,
                 vocab_size=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.ignore_bos = getattr(config.algo, 'ignore_bos', False)
        self.tokenizer = tokenizer
        if vocab_size is not None:
            self.vocab_size = vocab_size
        elif hasattr(tokenizer, 'vocab_size') and tokenizer.vocab_size is not None:
            # Use tokenizer.vocab_size (base vocab, before added tokens) then
            # account for any added special tokens so IDs like 126336 are valid.
            self.vocab_size = max(tokenizer.vocab_size,
                                  len(tokenizer) if tokenizer else 0)
        else:
            self.vocab_size = len(tokenizer)

        self.antithetic_sampling = config.training.antithetic_sampling
        self.parameterization = config.algo.parameterization

        # Backbone: always DITEdit for this repo
        self.backbone = DITEdit(config, vocab_size=self.vocab_size)

        self._pending_ema_state = None
        self.T = config.algo.T
        self.num_tokens = config.model.length
        self.softplus = torch.nn.Softplus()
        self.neg_infinity = -1_000_000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        self.noise = LogLinear()

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=config.eval.perplexity_batch_size)

        if config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(), decay=config.training.ema)
        else:
            self.ema = None

        self.lr = config.optim.lr
        self.sampling_eps = config.training.sampling_eps
        self.time_conditioning = config.algo.time_conditioning

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(),
                               self.noise.parameters())

    def _eval_mode(self):
        if self.ema and not self.config.eval.disable_ema:
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    # ------------------------------------------------------------------
    # Forward / model output
    # ------------------------------------------------------------------

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value=30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    def forward(self, xt, x_src_ids, sigma, sigma_prime=None,
                use_jvp_attn=False):
        """Process sigma scalars and call DITEdit backbone."""
        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(
                xt, x_src_ids, sigma, sigma_prime,
                use_jvp_attn=use_jvp_attn)
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def forward_no_softmax(self, xt, x_src_ids, tau, tau_prime=None, **kwargs):
        tau = self._process_sigma(tau)
        if tau_prime is not None:
            tau_prime = self._process_sigma(tau_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            return self.backbone(xt, x_src_ids, tau, tau_prime, **kwargs)

    # ------------------------------------------------------------------
    # Training & validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
            batch_idx % self.trainer.accumulate_grad_batches)

        x_src = batch['source_ids'].to(self.device)
        x_tgt = batch['target_ids'].to(self.device)
        valid_tokens = batch['attention_mask'].to(self.device)

        loss_obj = self._loss(x_src, x_tgt, valid_tokens,
                              current_accumulation_step)
        self.log('trainer/loss', loss_obj.loss.item(),
                 on_step=True, on_epoch=False, sync_dist=True)
        return loss_obj.loss

    def _loss(self, x_src, x_tgt, valid_tokens,
              current_accumulation_step=None):
        loss = self.loss({'source_ids': x_src, 'target_ids': x_tgt},
                         current_accumulation_step)
        assert loss.ndim == 2

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens
        return Loss(loss=token_nll, nlls=nlls,
                    prior_loss=0.0, num_tokens=num_tokens)

    def loss(self, batch, current_accumulation_step=None, **kwargs):
        raise NotImplementedError

    def on_train_epoch_start(self):
        self.metrics.reset()

    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()

    def validation_step(self, batch, batch_idx):
        x_src = batch['source_ids'].to(self.device)
        x_tgt = batch['target_ids'].to(self.device)
        valid_tokens = batch['attention_mask'].to(self.device)
        loss_obj = self._loss(x_src, x_tgt, valid_tokens)
        self.metrics.update_valid(loss_obj.nlls, loss_obj.prior_loss,
                                  loss_obj.num_tokens)
        # Log a few source→edit examples to W&B
        if batch_idx == 0:
            self._log_edit_samples(x_src, x_tgt)
        return loss_obj.loss

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(name=k, value=v.compute(), on_step=False,
                     on_epoch=True, sync_dist=True)
        self._train_mode()

    @torch.no_grad()
    def _log_edit_samples(self, x_src_ids, x_tgt_ids, n=4):
        try:
            n = min(n, x_src_ids.shape[0])
            pred_ids = self.generate_samples(x_src_ids[:n], num_steps=1)
            rows = []
            for i in range(n):
                src_text = self.tokenizer.decode(
                    x_src_ids[i], skip_special_tokens=True)
                tgt_text = self.tokenizer.decode(
                    x_tgt_ids[i], skip_special_tokens=True)
                pred_text = self.tokenizer.decode(
                    pred_ids[i], skip_special_tokens=True)
                rows.append([src_text, tgt_text, pred_text])
            if self.logger and hasattr(self.logger, 'experiment'):
                table = wandb.Table(
                    columns=['source', 'target', 'predicted'],
                    data=rows)
                self.logger.experiment.log({'val/edit_samples': table})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Optimiser / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)
        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler,
            optimizer=optimizer)
        return [optimizer], [{'scheduler': scheduler,
                               'interval': 'step',
                               'frequency': 1}]

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    # ------------------------------------------------------------------
    # Checkpoint helpers (mirror trainer_base.py)
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self._pending_ema_state = checkpoint.get('ema', None)
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress']['current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        if any('_orig_mod' in k for k in state_dict.keys()):
            state_dict = {k.replace('._orig_mod.', '.'): v
                          for k, v in state_dict.items()}
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith('teacher_model.')}
        ret = super().load_state_dict(state_dict, strict=False)
        if self.ema:
            ema_sd = getattr(self, '_pending_ema_state', None)
            if ema_sd is not None:
                try:
                    self.ema.load_state_dict(ema_sd)
                except Exception as e:
                    print(f"[WARNING] Failed to restore EMA: {e}")
            self._pending_ema_state = None
        return ret

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    # ------------------------------------------------------------------
    # Generation (must be overridden by subclass)
    # ------------------------------------------------------------------

    def generate_samples(self, src_ids, num_steps=1, **kwargs):
        raise NotImplementedError
