# Created by: Alex Myasnikov
# Date: May 14, 2026

"""GPT2 model factory + train/sample helpers for PatternBoost."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel

from algorithms.logger import log_structured
from algorithms.pattern_boost.tokenizer import FactoredWordTokenizer


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_positions: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 3
    lr: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50


@dataclass
class SampleGenConfig:
    n: int = 100
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    seed: Optional[int] = 42
    batch_size: int = 16


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(tokenizer: FactoredWordTokenizer, cfg: ModelConfig) -> GPT2LMHeadModel:
    gpt2_cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.n_positions,
        n_ctx=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(gpt2_cfg)


def _pad_batch(seqs: list[list[int]], pad_id: int, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = min(max(len(s) for s in seqs), max_len)
    n = len(seqs)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, s in enumerate(seqs):
        t = s[:L]
        input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        attn[i, :len(t)] = 1
    return input_ids, attn


def train(
    model: GPT2LMHeadModel,
    tokenizer: FactoredWordTokenizer,
    factor_words: list,
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    logger=None,
) -> None:
    if device is None:
        device = pick_device()
    model.to(device)
    model.train()

    max_len = model.config.n_positions
    pad_id = tokenizer.pad_token_id


    all_seqs = [tokenizer.encode(fw) for fw in factor_words]
    # for s in all_seqs:
    #     print(s)

    lens = [len(s) for s in all_seqs]
    n_over = sum(1 for L in lens if L > max_len)
    n_empty = sum(1 for L in lens if L == 0)
    if logger is not None and lens:
        sorted_lens = sorted(lens)
        def _pct(p):
            k = max(0, min(len(sorted_lens) - 1, int(round(p * (len(sorted_lens) - 1)))))
            return sorted_lens[k]
        stats = {
            'count': len(lens),
            'min': sorted_lens[0],
            'max': sorted_lens[-1],
            'mean': sum(lens) / len(lens),
            'p50': _pct(0.50), 'p90': _pct(0.90),
            'p95': _pct(0.95), 'p99': _pct(0.99),
            'max_len': max_len,
            'over_max_len': n_over,
            'over_max_len_pct': 100.0 * n_over / len(lens),
            'empty': n_empty,
        }
        log_structured(logger, 'info', event='pb_train_seq_lengths', **stats)
        if n_over / len(lens) > 0.01:
            log_structured(logger, 'warning', event='pb_train_truncation_warning',
                           dropped=n_over, total=len(lens), max_len=max_len,
                           message='more than 1% of sequences exceed n_positions and will be dropped — '
                                   'increase model.n_positions to avoid losing data')

    all_seqs = [s for s in all_seqs if 0 < len(s) <= max_len]
    if not all_seqs:
        raise ValueError('no training sequences fit within model context')

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n = len(all_seqs)
    step = 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n).tolist()
        for i in range(0, n, cfg.batch_size):
            batch_seqs = [all_seqs[j] for j in perm[i:i + cfg.batch_size]]
            input_ids, attn = _pad_batch(batch_seqs, pad_id, max_len)
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            labels = input_ids.clone()
            labels[attn == 0] = -100

            optim.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            if logger is not None and step % cfg.log_every == 0:
                log_structured(
                    logger, 'info',
                    event='pb_train_step',
                    epoch=epoch, step=step, loss=float(loss.item()),
                )
            step += 1


@torch.no_grad()
def sample(
    model: GPT2LMHeadModel,
    tokenizer: FactoredWordTokenizer,
    cfg: SampleGenConfig,
    device: Optional[torch.device] = None,
) -> list:
    if device is None:
        device = pick_device()
    model.to(device)
    model.eval()

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

    out_fws = []
    remaining = cfg.n
    while remaining > 0:
        bs = min(cfg.batch_size, remaining)
        prompt = torch.full((bs, 1), bos, dtype=torch.long, device=device)
        gen = model.generate(
            prompt,
            max_length=cfg.max_length,
            do_sample=True,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            pad_token_id=pad,
            eos_token_id=eos,
        )
        for row in gen:
            fw = tokenizer.try_decode(row.tolist())
            if fw is not None:
                out_fws.append(fw)
        remaining -= bs
    return out_fws


def save_checkpoint(model: GPT2LMHeadModel, path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))


def load_checkpoint(path) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(str(path))
