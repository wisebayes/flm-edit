"""FLMEdit and FMLMEdit algorithm classes for one-step text editing.

Stage 1 (FLMEdit): learns a single-time denoiser D_t(x_t | x_src).
Stage 2 (FMLMEdit): distills into a two-time flow map δ_{s,t} via PSD,
                    enabling one-step editing: x̂_tgt = δ_{0,1}(x_src).

Follows the structure of algo.py in https://github.com/david3684/flm but
replaces Gaussian noise with the source-text one-hot as the t=0 endpoint.
"""
import collections
import copy

import torch
import torch.nn.functional as F

import trainer_edit
import utils
import models
from models.dit_edit import DITEdit


def stopgrad(x):
    return x.detach()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FLMEditBase(trainer_edit.TrainerEditBase):
    """Base for both FLMEdit (stage 1) and FMLMEdit (stage 2).

    The key departure from the unconditional FLMBase: corrupt_continuous
    interpolates between source and target one-hots rather than between
    Gaussian noise and a one-hot.  The interpolant stays on the simplex
    for all t ∈ [0,1].
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    # ------------------------------------------------------------------
    # Time reparameterisation (reuse upstream LUTs unchanged)
    # ------------------------------------------------------------------

    def _tau_to_t(self, tau):
        return utils.alpha_to_gamma(tau, self.lut_a2g)

    def _t_to_tau(self, t):
        return utils.gamma_to_alpha(t, self.lut_g2a)

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]
        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t = t[:batch_dim]
        return t

    # ------------------------------------------------------------------
    # Interpolant
    # ------------------------------------------------------------------

    def corrupt_continuous(self, x_src_ids, x_tgt_ids, t):
        """Source→target linear interpolant at time t.

        Both endpoints are one-hot simplex points, so x_t stays on the
        simplex for all t — unlike the unconditional FLM where x_0 is
        Gaussian (off-simplex).

        Args:
            x_src_ids: (B, L) integer source token IDs
            x_tgt_ids: (B, L) integer target token IDs
            t:         (B,) interpolation time in [0, 1]
        Returns:
            x_t:          (B, L, V) interpolated continuous embedding
            target_data:  (B, L, V) one-hot target (for loss)
        """
        t = t.unsqueeze(-1).unsqueeze(-1)                    # (B,1,1)
        x_src = F.one_hot(x_src_ids, self.vocab_size).float()
        x_tgt = F.one_hot(x_tgt_ids, self.vocab_size).float()
        x_t = (1.0 - t) * x_src + t * x_tgt
        return x_t, x_tgt

    # ------------------------------------------------------------------
    # Checkpoint helpers (strip teacher from saved state)
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = self._filter_checkpoint_state_dict(
                checkpoint['state_dict'])
        super().on_load_checkpoint(checkpoint)

    def _filter_checkpoint_state_dict(self, state_dict):
        new_sd = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('teacher'):
                continue
            new_key = k.replace('._orig_mod.', '.')
            new_sd[new_key] = v
        return new_sd

    # ------------------------------------------------------------------
    # Teacher loading helpers (for FMLMEdit; reuse pattern from FLMBase)
    # ------------------------------------------------------------------

    def _extract_ema_state_dict(self, model, checkpoint):
        ema_state = checkpoint.get('ema', None)
        if not ema_state:
            print("Warning: No EMA found in teacher checkpoint, "
                  "using regular state_dict")
            return {k.replace('backbone.', '').replace('._orig_mod.', ''): v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith('backbone.')}
        new_sd = collections.OrderedDict()
        shadow_params = ema_state['shadow_params']
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        min_len = min(len(shadow_params), len(param_names))
        for name, val in zip(param_names[:min_len], shadow_params[:min_len]):
            new_sd[name] = val
        return new_sd

    def _load_teacher_model(self, path):
        """Load a frozen FLMEdit checkpoint as teacher for distillation."""
        print(f"Loading teacher model from: {path}")
        # Build teacher with single-time conditioning (no double_temb)
        saved_double = self.config.algo.double_temb
        self.config.algo.double_temb = False
        model = DITEdit(self.config, vocab_size=self.vocab_size)
        self.config.algo.double_temb = saved_double

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _copy_teacher_weights_to_student(self, teacher_dict):
        with torch.no_grad():
            student_dict = self.backbone.state_dict()
            for name, param in teacher_dict.items():
                if name in student_dict:
                    student_dict[name].copy_(param)
            if (hasattr(self.backbone, 'sigma_map_prime')
                    and self.backbone.sigma_map_prime is not None):
                for name, param in self.backbone.sigma_map_prime.named_parameters():
                    if 'mlp.2' in name:
                        param.zero_()


# ---------------------------------------------------------------------------
# Stage 1: FLMEdit
# ---------------------------------------------------------------------------

class FLMEdit(FLMEditBase):
    """Stage 1: single-time denoiser D_t(x_t | x_src).

    Training: cross-entropy against the one-hot target (eq. 12 in the paper).
    Inference: multi-step Euler ODE integration from source to edited target.
    """

    def loss(self, batch, current_accumulation_step=None, **kwargs):
        x_src_ids = batch['source_ids']    # (B, L)
        x_tgt_ids = batch['target_ids']    # (B, L)
        B = x_src_ids.shape[0]

        tau_t = self._sample_t_interval(B, current_accumulation_step)
        t = self._tau_to_t(tau_t)
        x_t, target_data = self.corrupt_continuous(x_src_ids, x_tgt_ids, t)

        log_probs = self.forward(x_t, x_src_ids, tau_t)   # (B, L, V)

        # Optionally upweight positions where source differs from target
        loss = -(target_data * log_probs).sum(dim=-1)      # (B, L)
        if getattr(self.config.algo, 'upweight_changed_tokens', False):
            changed = (x_src_ids != x_tgt_ids).float()
            weight = 1.0 + changed
            loss = loss * weight

        self.log('loss', loss.mean(), prog_bar=True)
        return loss

    def nll(self, batch, current_accumulation_step=None):
        return self.loss(batch, current_accumulation_step)

    @torch.no_grad()
    def generate_samples(self, src_ids, num_steps=None, eps=1e-5):
        """Multi-step Euler ODE from source to edited target.

        At num_steps=1 this is a single denoiser call (approximate).
        Use num_steps ≥ 16 for high quality during stage-1 eval.
        """
        if num_steps is None:
            steps_cfg = self.config.sampling.steps
            num_steps = steps_cfg[0] if isinstance(steps_cfg, (list, tuple)) else steps_cfg

        B, L = src_ids.shape
        V = self.vocab_size
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        # Start at source (t=0 endpoint of interpolant)
        z = F.one_hot(src_ids, V).float()

        for i in range(num_steps):
            tau_curr = tau_vals[i].expand(B)
            tau_next = tau_vals[i + 1].expand(B)
            t_curr = self._tau_to_t(tau_curr)
            dt = self._tau_to_t(tau_next) - t_curr

            log_probs = self.forward(z, src_ids, tau_curr)
            x_1_pred = log_probs.exp()

            if i == num_steps - 1:
                z = x_1_pred
                break

            v = (x_1_pred - z) / (1.0 - t_curr.view(-1, 1, 1) + eps)
            z = z + dt.view(-1, 1, 1) * v

        return z.argmax(dim=-1)    # (B, L)


# ---------------------------------------------------------------------------
# Stage 2: FMLMEdit
# ---------------------------------------------------------------------------

class FMLMEdit(FLMEditBase):
    """Stage 2: two-time flow map δ_{s,t}(x | x_src).

    Distilled from FLMEdit via Progressive Semigroup Distillation (PSD).
    One-step inference: x̂_tgt = δ_{0,1}(x_src | x_src).
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.teacher_model = None

    def _validate_configuration(self):
        assert self.config.algo.double_temb is True, \
            "FMLMEdit requires double_temb=True for two-time conditioning"
        assert isinstance(self.config.algo.diagonal_fraction, float), \
            "diagonal_fraction must be a float"
        assert 0 <= self.config.algo.diagonal_fraction <= 1
        assert self.config.algo.distillation_method in ('PSD', 'LSD', 'ESD'), \
            "distillation_method must be PSD, LSD, or ESD"

    def setup(self, stage: str):
        if self.teacher_model is None:
            self._initialize_teacher()
        if stage == 'fit' and not self._is_resuming:
            print(">>> Initializing student from teacher...")
            self._initialize_student_from_teacher()
        elif self._is_resuming:
            print(">>> Skipping student init (resuming from checkpoint).")

    def _initialize_teacher(self):
        path = self.config.algo.teacher_path
        if not path:
            print("No teacher_path specified; skipping teacher init")
            return
        self.teacher_model = self._load_teacher_model(path)

    def _initialize_student_from_teacher(self):
        if (self.teacher_model is None
                or not self.config.algo.initialize_student_from_teacher):
            return
        self._copy_teacher_weights_to_student(
            self.teacher_model.state_dict())

    def forward_with_ema(self, *args, **kwargs):
        assert self.ema is not None, "EMA must be available"
        self.ema.store(self._get_parameters())
        self.ema.copy_to(self._get_parameters())
        try:
            with torch.no_grad():
                self.backbone.eval()
                return self.forward(*args, **kwargs)
        finally:
            self.ema.restore(self._get_parameters())
            self.backbone.train()

    def teacher_forward(self, xt, x_src_ids, tau):
        """Single-time forward pass through the frozen teacher."""
        sigma = self._process_sigma(tau)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type,
                                    dtype=torch.float32):
                logits = self.teacher_model(xt, x_src_ids, sigma)
        return self._process_model_output(logits, xt, sigma)

    def _d_tau_by_d_t(self, t):
        return utils.d_alpha_by_d_gamma(t, self.lut_g2a)

    def _sample_multi_t_interval(self, n, accum_step, num_per_sample,
                                 t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        total_n = (self.config.loader.global_batch_size * num_per_sample
                   if accum_step is not None else n * num_per_sample)
        _eps = torch.rand(total_n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(total_n, device=self.device) / total_n
            _eps = (_eps / total_n + offset) % 1
            _eps = _eps[torch.randperm(total_n, device=self.device)]
        t_all = (t_max - t_min) * _eps + t_min
        if accum_step is not None:
            t_all = t_all.view(-1, num_per_sample)
            t_all = t_all.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t_all = t_all.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t_all = t_all.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t_all = t_all[:n]
        else:
            t_all = t_all.view(n, num_per_sample)
        t_sorted, _ = torch.sort(t_all, dim=-1)
        return [t_sorted[:, i] for i in range(num_per_sample)]

    def _get_split_indices(self, n, accum_step, ratios):
        assert sum(ratios) < 0.999
        if accum_step is not None:
            n_total = self.config.loader.global_batch_size
            num_nodes = self.trainer.num_nodes
            num_devices = self.trainer.num_devices
            accum_batches = self.trainer.accumulate_grad_batches
            node_rank = self.trainer.node_rank
            local_rank = self.trainer.local_rank
            current_accum = accum_step
        else:
            n_total = n
            num_nodes = num_devices = accum_batches = 1
            node_rank = local_rank = current_accum = 0

        total_chunks = num_nodes * num_devices * accum_batches
        chunk_idx = (node_rank * num_devices * accum_batches
                     + local_rank * accum_batches + current_accum)
        num_categories = len(ratios) + 1
        global_counts = [0] * num_categories
        net_ratio, prev_num = 0, 0
        for i, ratio in enumerate(ratios):
            net_ratio += ratio
            num_a = int(n_total * net_ratio)
            global_counts[i] = num_a - prev_num
            prev_num = num_a
        global_counts[-1] = n_total - prev_num

        local_counts = []
        for cnt in global_counts:
            base, rem = divmod(cnt, total_chunks)
            local_counts.append(base + (1 if chunk_idx < rem else 0))

        local_size = sum(local_counts)
        local_assignments = torch.empty(local_size, device=self.device,
                                        dtype=torch.long)
        offset = 0
        for i, cnt in enumerate(local_counts):
            local_assignments[offset:offset + cnt] = i
            offset += cnt

        seed = self.global_step * total_chunks + chunk_idx
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        perm = torch.randperm(local_size, device=self.device, generator=g)
        local_assignments = local_assignments[perm][:n]
        return [(local_assignments == i).nonzero(as_tuple=True)[0]
                for i in range(num_categories)]

    # ------------------------------------------------------------------
    # PSD loss (eq. 22–23 in the paper)
    # ------------------------------------------------------------------

    def loss(self, batch, current_accumulation_step=None, **kwargs):
        x_src_ids = batch['source_ids']
        x_tgt_ids = batch['target_ids']
        B, L = x_src_ids.shape

        tau_diag = self._sample_t_interval(B, current_accumulation_step)
        set_midpoint = getattr(self.config.algo, 'set_midpoint', 'midpoint')

        if self.config.algo.offdiagonal_sampling == 'uniform_st':
            tau_s_offdiag, tau_t_offdiag = self._sample_multi_t_interval(
                B, current_accumulation_step, 2)
        else:  # uniform_diff
            tau_d = self._sample_t_interval(B, current_accumulation_step)
            tau_s_offdiag = self._sample_t_interval(
                B, current_accumulation_step) * (1 - tau_d)
            tau_t_offdiag = tau_s_offdiag + tau_d

        idx_diag, idx_offdiag_bndry, idx_offdiag = self._get_split_indices(
            B, current_accumulation_step,
            ratios=(self.config.algo.diagonal_fraction,
                    (1.0 - self.config.algo.diagonal_fraction)
                    * (1.0 / self.config.algo.boundary_prob)))

        tau_s = torch.zeros(B, device=self.device)
        tau_t = torch.zeros(B, device=self.device)
        tau_s[idx_diag] = tau_diag[idx_diag]
        tau_s[idx_offdiag_bndry] = 0.0
        tau_s[idx_offdiag] = tau_s_offdiag[idx_offdiag]
        tau_t[idx_diag] = tau_diag[idx_diag]
        tau_t[idx_offdiag_bndry] = 1.0
        tau_t[idx_offdiag] = tau_t_offdiag[idx_offdiag]
        tau_s = tau_s.clamp(0.0, 1.0)
        tau_t = tau_t.clamp(0.0, 1.0)

        # Merge off-diag + boundary indices
        idx_offdiag = torch.cat([idx_offdiag, idx_offdiag_bndry])
        has_diag = idx_diag.numel() > 0
        has_offdiag = idx_offdiag.numel() > 0

        if set_midpoint == 'midpoint':
            tau_u = 0.5 * (tau_s + tau_t)
        else:
            tau_u = tau_s + torch.rand_like(tau_s) * (tau_t - tau_s)

        s = self._tau_to_t(tau_s)
        u = self._tau_to_t(tau_u)
        t = self._tau_to_t(tau_t)

        x_s, target_data = self.corrupt_continuous(x_src_ids, x_tgt_ids, s)

        # ---- diagonal: anchor to teacher or one-hot ----
        if has_diag:
            if self.teacher_model is not None:
                on_diag_target = self.teacher_forward(
                    x_s[idx_diag], x_src_ids[idx_diag],
                    tau_s[idx_diag]).exp()
            else:
                on_diag_target = target_data[idx_diag]
            on_diag_target = stopgrad(on_diag_target)

        # ---- full two-time forward pass ----
        log_D_st = self.forward(x_s, x_src_ids, tau_s, tau_t)  # (B, L, V)

        # ---- off-diagonal: PSD semigroup (eq. 22) ----
        if has_offdiag:
            with torch.no_grad():
                x_s_od = x_s[idx_offdiag]
                src_od = x_src_ids[idx_offdiag]
                s_od = s[idx_offdiag].view(-1, 1, 1)
                u_od = u[idx_offdiag].view(-1, 1, 1)
                t_od = t[idx_offdiag].view(-1, 1, 1)
                tau_s_od = tau_s[idx_offdiag]
                tau_u_od = tau_u[idx_offdiag]
                tau_t_od = tau_t[idx_offdiag]

                _fwd = (self.forward_with_ema
                        if getattr(self.config.algo,
                                   'use_ema_for_psd_target', False)
                        else self.forward)

                D_su = _fwd(x_s_od, src_od, tau_s_od, tau_u_od).exp()
                # Advance along flow map: eq. 19 in paper
                X_su = ((1 - u_od) / (1 - s_od + 1e-8)) * x_s_od \
                     + ((u_od - s_od) / (1 - s_od + 1e-8)) * D_su
                D_ut = _fwd(X_su, src_od, tau_u_od, tau_t_od).exp()
                # Semigroup combination: eq. 22
                lam = ((1 - t_od) * (u_od - s_od)
                       / ((1 - u_od) * (t_od - s_od) + 1e-8))
                offdiag_target = stopgrad(lam * D_su + (1 - lam) * D_ut)

        # ---- compute losses ----
        if has_diag:
            use_mse = getattr(self.config.algo, 'use_mse_loss_psd', False)
            if not use_mse:
                diag_loss = -(on_diag_target * log_D_st[idx_diag]).sum(dim=-1)
            else:
                diag_loss = F.mse_loss(log_D_st[idx_diag].exp(),
                                       on_diag_target,
                                       reduction='none').sum(dim=-1)
        else:
            diag_loss = x_s.new_empty((0, L))

        if has_offdiag:
            use_mse = getattr(self.config.algo, 'use_mse_loss_psd', False)
            if not use_mse:
                offdiag_loss = -(offdiag_target
                                 * log_D_st[idx_offdiag]).sum(dim=-1)
            else:
                offdiag_loss = F.mse_loss(log_D_st[idx_offdiag].exp(),
                                          offdiag_target,
                                          reduction='none').sum(dim=-1)
            if getattr(self.config.algo, 'rescale_offdiag_loss_psd', False):
                offdiag_loss = offdiag_loss * (
                    (t_od - s_od) / (1 - s_od + 1e-8)
                ).view(-1, 1).pow(2)
        else:
            offdiag_loss = x_s.new_empty((0, L))

        loss = torch.zeros(B, L, device=self.device)
        if has_diag:
            loss[idx_diag] = diag_loss
        if has_offdiag:
            loss[idx_offdiag] = offdiag_loss

        self.log('loss', loss.mean(), prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples(self, src_ids, num_steps=1, eps=1e-5):
        """One-step (or few-step) editing using the two-time flow map.

        At num_steps=1: single forward pass → x̂_tgt = δ_{0,1}(x_src | x_src).
        """
        B, L = src_ids.shape
        V = self.vocab_size
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = F.one_hot(src_ids, V).float()   # start at source (t=0)

        for i in range(num_steps):
            tau_s = tau_vals[i].expand(B)
            tau_t = tau_vals[i + 1].expand(B)
            t_s = self._tau_to_t(tau_s).view(-1, 1, 1)
            t_t = self._tau_to_t(tau_t).view(-1, 1, 1)

            log_D = self.forward(z, src_ids, tau_vals[i].expand(B),
                                 tau_vals[i + 1].expand(B))
            D = log_D.exp()

            if i == num_steps - 1:
                z = D
                break

            # Flow map update rule: eq. 19 in paper
            z = ((1 - t_t) / (1 - t_s + eps)) * z \
              + ((t_t - t_s) / (1 - t_s + eps)) * D

        return z.argmax(dim=-1)    # (B, L)
