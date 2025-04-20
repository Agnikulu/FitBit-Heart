# src/models.py
"""Model definitions – forecasting backbone + classifier.
All structural hyper‑parameters come from cfg.model (hierarchical YAML).
"""
from __future__ import annotations

from typing import Sequence, Tuple
import torch
import torch.nn as nn

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _make_cnn(channels: Sequence[int], dropout: float) -> nn.Sequential:
    """Construct a 1‑D CNN stack given a list of output channels."""
    layers, in_ch = [], 1
    for out_ch in channels:
        layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                   nn.BatchNorm1d(out_ch),
                   nn.ReLU(),
                   nn.Dropout(dropout)]
        in_ch = out_ch
    return nn.Sequential(*layers)

# ------------------------------------------------------------------ #
class SSLForecastingModel(nn.Module):
    """Cross‑modal self‑supervised forecaster (BPM ⇄ Steps)."""
    def __init__(self,
                 *,
                 window_size: int,
                 predict_windows: int,
                 cfg_model: dict):
        super().__init__()
        self.window_size     = window_size
        self.predict_windows = predict_windows

        # pull hyper‑params from the YAML sub‑dict
        embed_dim   = cfg_model.get("embed_dim",   256)
        cnn_ch      = cfg_model.get("cnn_channels", [32, 64])
        rnn_hidden  = cfg_model.get("rnn_hidden",  128)
        rnn_layers  = cfg_model.get("rnn_layers",    2)
        attn_heads  = cfg_model.get("attn_heads",    1)
        dropout     = cfg_model.get("dropout",     0.3)

        # — encoders for the *past* windows —
        self.bpm_cnn   = _make_cnn(cnn_ch, dropout)
        self.steps_cnn = _make_cnn(cnn_ch, dropout)
        self.bpm_rnn   = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)
        self.steps_rnn = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)

        # — per‑window aggregators for the *current* windows —
        self.agg_current_bpm   = nn.Sequential(nn.Linear(window_size, 16), nn.ReLU(), nn.Dropout(dropout))
        self.agg_current_steps = nn.Sequential(nn.Linear(window_size, 16), nn.ReLU(), nn.Dropout(dropout))
        self.curr_proj         = nn.Linear(16, embed_dim)

        # positional embedding for the sequence [past_summary, P×current_window]
        self.pos_emb = nn.Embedding(predict_windows + 1, embed_dim)

        # attention + FFN stacks (one per prediction path)
        def _attn_block() -> Tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
            attn = nn.MultiheadAttention(embed_dim, attn_heads, batch_first=True)
            ln1  = nn.LayerNorm(embed_dim)
            ffn  = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(embed_dim, embed_dim), nn.Dropout(dropout))
            ln2  = nn.LayerNorm(embed_dim)
            return attn, ln1, ffn, ln2

        (self.attn_bpm,   self.ln1_bpm,   self.ffn_bpm,   self.ln2_bpm)   = _attn_block()
        (self.attn_steps, self.ln1_steps, self.ffn_steps, self.ln2_steps) = _attn_block()

        # decoders | token → future window (6‑h) prediction
        self.fusion_bpm   = nn.Linear(embed_dim, window_size)
        self.fusion_steps = nn.Linear(embed_dim, window_size)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _extract_past_summary(self, bpm_in: torch.Tensor, steps_in: torch.Tensor) -> torch.Tensor:
        """CNN → GRU → last hidden for each modality, concat → 256‑d summary."""
        B = bpm_in.size(0)

        def _path(x: torch.Tensor, cnn: nn.Module, rnn: nn.Module) -> torch.Tensor:
            v = x.view(B, -1).unsqueeze(1)      # [B,1,L]
            v = cnn(v).permute(0, 2, 1)         # [B,L,channels]
            h, _ = rnn(v)                       # GRU
            return h[:, -1, :]                  # last hidden

        h_bpm   = _path(bpm_in,   self.bpm_cnn,   self.bpm_rnn)
        h_steps = _path(steps_in, self.steps_cnn, self.steps_rnn)
        return torch.cat([h_bpm, h_steps], dim=1)  # [B,256]

    def _make_tokens(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Project list of shape‑[B,16] tensors → single [B,P,embed] tensor."""
        return torch.stack([self.curr_proj(f) for f in feats], dim=1)

    # ------------------------------------------------------------------ #
    def forward(self,
                bpm_in: torch.Tensor,     # [B, input_windows,  W]
                steps_in: torch.Tensor,   # [B, input_windows,  W]
                cur_bpm: torch.Tensor,    # [B, P,             W]
                cur_steps: torch.Tensor   # [B, P,             W]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P, W = bpm_in.size(0), self.predict_windows, self.window_size

        # 1) past summary token (256‑d)
        past_summary = self._extract_past_summary(bpm_in, steps_in)  # [B,256]

        # 2) path‑specific current‑window features → tokens
        feats_bpm   = [self.agg_current_steps(cur_steps[:, idx, :]) for idx in range(P)]
        feats_steps = [self.agg_current_bpm(  cur_bpm[:,   idx, :]) for idx in range(P)]

        tok_bpm   = self._make_tokens(feats_bpm)    # [B,P,embed]
        tok_steps = self._make_tokens(feats_steps)  # [B,P,embed]

        def _add_pos(tokens: torch.Tensor) -> torch.Tensor:
            seq = torch.cat([past_summary.unsqueeze(1), tokens], dim=1)  # [B,P+1,embed]
            idx = torch.arange(P + 1, device=seq.device)
            return seq + self.pos_emb(idx)[None, :, :]

        seq_bpm   = _add_pos(tok_bpm)
        seq_steps = _add_pos(tok_steps)

        # 3) Transformer (per path)
        def _xformer(seq: torch.Tensor,
                     attn: nn.Module, ln1: nn.Module,
                     ffn: nn.Module, ln2: nn.Module) -> torch.Tensor:
            attn_out, _ = attn(seq, seq, seq)
            x = ln1(seq + attn_out)
            ffn_out = ffn(x)
            return ln2(x + ffn_out)

        res_bpm   = _xformer(seq_bpm,   self.attn_bpm,   self.ln1_bpm,   self.ffn_bpm,   self.ln2_bpm)
        res_steps = _xformer(seq_steps, self.attn_steps, self.ln1_steps, self.ffn_steps, self.ln2_steps)

        # 4) decode (drop index‑0 = past summary)
        out_bpm   = self.fusion_bpm(res_bpm[:,   1:, :])   # [B,P,W]
        out_steps = self.fusion_steps(res_steps[:, 1:, :]) # [B,P,W]
        return out_bpm, out_steps

# ------------------------------------------------------------------ #
class PersonalizedForecastingModel(SSLForecastingModel):
    """Same architecture – used for fine‑tuning per user."""
    pass

# ------------------------------------------------------------------ #
class DrugClassifier(nn.Module):
    """Binary classifier (use / crave) built atop the CNN+GRU encoder."""
    def __init__(self, *, window_size: int, cfg_model: dict):
        super().__init__()
        dropout    = cfg_model.get("dropout",     0.3)
        cnn_ch     = cfg_model.get("cnn_channels", [32, 64])
        rnn_hidden = cfg_model.get("rnn_hidden",  128)
        rnn_layers = cfg_model.get("rnn_layers",    2)

        self.bpm_cnn   = _make_cnn(cnn_ch, dropout)
        self.steps_cnn = _make_cnn(cnn_ch, dropout)
        self.bpm_rnn   = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)
        self.steps_rnn = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(rnn_hidden, 1))

    # ------------------------------------------------------------------ #
    def forward(self, bpm_in: torch.Tensor, steps_in: torch.Tensor) -> torch.Tensor:
        B = bpm_in.size(0)

        def _path(x: torch.Tensor, cnn: nn.Module, rnn: nn.Module) -> torch.Tensor:
            v = x.view(B, -1).unsqueeze(1)     # [B,1,window_size]
            v = cnn(v).permute(0, 2, 1)        # [B,L,chan]
            h, _ = rnn(v)
            return h[:, -1, :]                 # last hidden of GRU

        z_bpm   = _path(bpm_in,   self.bpm_cnn,   self.bpm_rnn)
        z_steps = _path(steps_in, self.steps_cnn, self.steps_rnn)
        fused   = torch.cat([z_bpm, z_steps], dim=1)  # [B, 2*rnn_hidden]
        return self.classifier(fused).squeeze(-1)     # logits

# ------------------------------------------------------------------ #
# Utility: partial unfreeze                                           #
# ------------------------------------------------------------------ #

def partially_unfreeze_backbone(model: nn.Module, unfreeze_ratio: float = 0.3) -> None:
    """Unfreeze the last *unfreeze_ratio* fraction of parameters in‑place."""
    params = list(model.named_parameters())
    cutoff = int(len(params) * (1.0 - unfreeze_ratio))
    for idx, (_, p) in enumerate(params):
        p.requires_grad = idx >= cutoff