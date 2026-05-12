"""Entry point for flm-edit training and evaluation.

Adapted from main.py in https://github.com/david3684/flm.
Algorithm dispatch extended with flm_edit and fmlm_edit.
"""
import functools
import json
import os
import uuid

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import wandb

torch.load = functools.partial(torch.load, weights_only=False)
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,
])

import algo_edit
import utils
from data.edit_dataset import EditDataset
from data.maskdisc_dataset import MaskDiscEditDataset
from data.gpt2_edit_dataset import GPT2EditDataset
from transformers import AutoTokenizer

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count',
                                           torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)

ALGORITHM_MAP = {
    'flm_edit': algo_edit.FLMEdit,
    'fmlm_edit': algo_edit.FMLMEdit,
    'flm_edit_finetune': algo_edit.FLMEditFinetune,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_tokenizer(config):
    tok = AutoTokenizer.from_pretrained(
        config.data.tokenizer_name_or_path,
        cache_dir=getattr(config.data, 'cache_dir', None))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _make_dataset(config, tokenizer, split: str):
    """Instantiate the correct dataset class based on config.data.dataset_type."""
    path = config.data.train_path if split == 'train' else config.data.valid_path
    dtype = getattr(config.data, 'dataset_type', 'edit')

    if dtype == 'maskdisc':
        context_key = getattr(config.data, 'context_key', None)
        ctx_tok = tokenizer if context_key else None
        return MaskDiscEditDataset(
            path=path,
            max_length=config.data.max_length,
            context_key=context_key,
            tokenizer=ctx_tok,
            context_max_length=getattr(config.data, 'context_max_length', 512),
        )
    elif dtype == 'gpt2_edit':
        return GPT2EditDataset(
            path=path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            masking_strategy=getattr(config.data, 'masking_strategy', 'random'),
            use_context=getattr(config.data, 'use_context', False),
            context_key=getattr(config.data, 'context_key', 'dialogue'),
            context_max_length=getattr(config.data, 'context_max_length', 256),
        )
    else:
        return EditDataset(
            path, tokenizer,
            max_length=config.data.max_length,
            instruction_max_length=getattr(config.data, 'instruction_max_length', 64),
        )


def get_dataloaders(config, tokenizer):
    from torch.utils.data import DataLoader
    train_ds = _make_dataset(config, tokenizer, 'train')
    valid_ds = _make_dataset(config, tokenizer, 'valid')
    train_dl = DataLoader(
        train_ds, batch_size=config.loader.batch_size,
        shuffle=True, num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory, drop_last=True)
    valid_dl = DataLoader(
        valid_ds, batch_size=config.loader.eval_batch_size,
        shuffle=False, num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory)
    return train_dl, valid_dl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@L.pytorch.utilities.rank_zero_only
def _print_config(config, resolve=True, save_cfg=True):
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)
    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)
        section = config.get(field)
        content = (omegaconf.OmegaConf.to_yaml(section, resolve=resolve)
                   if isinstance(section, omegaconf.DictConfig)
                   else str(section))
        branch.add(rich.syntax.Syntax(content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open('{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


def _load_from_checkpoint(model_cls, config, tokenizer):
    return model_cls.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer, config=config,
        weights_only=False)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(model_cls, config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wid = config.wandb.get('id')
        if not wid or len(str(wid)) > 16:
            wid = str(uuid.uuid4().hex[:8])
        config.wandb.id = wid
        if config.wandb.get('name'):
            config.wandb.name = f"{config.wandb.name}_{wid}"
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            **config.wandb)

    ckpt_path = None
    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path

    callbacks = []
    if 'callbacks' in config:
        for _, cb in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb))

    train_dl, valid_dl = get_dataloaders(config, tokenizer)

    if config.training.finetune_path:
        assert utils.fsspec_exists(config.training.finetune_path)
        model = model_cls.load_from_checkpoint(
            config.training.finetune_path,
            tokenizer=tokenizer, config=config, weights_only=False)
    else:
        model = model_cls(config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_dl, valid_dl, ckpt_path=ckpt_path)


# ---------------------------------------------------------------------------
# Sample eval
# ---------------------------------------------------------------------------

def _generate_samples(model_cls, config, logger, tokenizer):
    logger.info('Starting Edit Sample Eval.')
    model = _load_from_checkpoint(model_cls, config, tokenizer)
    if config.eval.disable_ema:
        model.ema = None

    steps_cfg = config.sampling.steps
    step_list = list(steps_cfg) if hasattr(steps_cfg, '__iter__') else [steps_cfg]

    all_results = {}
    for num_steps in step_list:
        samples = []
        _, valid_dl = get_dataloaders(config, tokenizer)
        for batch in valid_dl:
            src_ids = batch['source_ids'].cuda()
            with torch.no_grad():
                pred_ids = model.generate_samples(src_ids, num_steps=num_steps)
            for i in range(src_ids.shape[0]):
                samples.append({
                    'source': tokenizer.decode(src_ids[i],
                                               skip_special_tokens=True),
                    'predicted': tokenizer.decode(pred_ids[i],
                                                  skip_special_tokens=True),
                })
            if len(samples) >= config.loader.eval_batch_size:
                break
        all_results[f'steps_{num_steps}'] = samples
        print(f'Steps={num_steps}: generated {len(samples)} samples')

    with fsspec.open(config.eval.generated_samples_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print('Saved to', config.eval.generated_samples_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = get_tokenizer(config)

    algo_name = config.algo.name
    if algo_name not in ALGORITHM_MAP:
        raise ValueError(f'Unknown algorithm: {algo_name}. '
                         f'Available: {list(ALGORITHM_MAP.keys())}')
    model_cls = ALGORITHM_MAP[algo_name]

    if config.mode == 'sample_eval':
        _generate_samples(model_cls, config, logger, tokenizer)
    else:
        _train(model_cls, config, logger, tokenizer)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
