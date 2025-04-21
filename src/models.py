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
        layers += [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        in_ch = out_ch
    return nn.Sequential(*layers)

# ------------------------------------------------------------------ #
class SSLForecastingModel(nn.Module):
    """Cross‑modal self‑supervised forecaster (BPM ⇄ Steps)."""
    def __init__(
        self,
        *,
        window_size: int,
        predict_windows: int,
        cfg_model: dict
    ):
        super().__init__()
        self.window_size = window_size
        self.predict_windows = predict_windows

        embed_dim = cfg_model.get("embed_dim", 256)
        cnn_ch = cfg_model.get("cnn_channels", [32, 64])
        rnn_hidden = cfg_model.get("rnn_hidden", 128)
        rnn_layers = cfg_model.get("rnn_layers", 2)
        attn_heads = cfg_model.get("attn_heads", 1)
        dropout = cfg_model.get("dropout", 0.3)

        self.bpm_cnn = _make_cnn(cnn_ch, dropout)
        self.steps_cnn = _make_cnn(cnn_ch, dropout)
        self.bpm_rnn = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)
        self.steps_rnn = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)

        self.agg_current_bpm = nn.Sequential(
            nn.Linear(window_size, 16), nn.ReLU(), nn.Dropout(dropout)
        )
        self.agg_current_steps = nn.Sequential(
            nn.Linear(window_size, 16), nn.ReLU(), nn.Dropout(dropout)
        )
        self.curr_proj = nn.Linear(16, embed_dim)

        self.pos_emb = nn.Embedding(predict_windows + 1, embed_dim)

        def _attn_block():
            attn = nn.MultiheadAttention(embed_dim, attn_heads, batch_first=True)
            ln1 = nn.LayerNorm(embed_dim)
            ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(dropout)
            )
            ln2 = nn.LayerNorm(embed_dim)
            return attn, ln1, ffn, ln2

        self.attn_bpm, self.ln1_bpm, self.ffn_bpm, self.ln2_bpm = _attn_block()
        self.attn_steps, self.ln1_steps, self.ffn_steps, self.ln2_steps = _attn_block()

        self.fusion_bpm = nn.Linear(embed_dim, window_size)
        self.fusion_steps = nn.Linear(embed_dim, window_size)

    def _extract_past_summary(
        self, bpm_in: torch.Tensor, steps_in: torch.Tensor
    ) -> torch.Tensor:
        B = bpm_in.size(0)

        def _path(x, cnn, rnn):
            v = x.view(B, -1).unsqueeze(1)
            v = cnn(v).permute(0, 2, 1)
            h, _ = rnn(v)
            return h[:, -1, :]

        h_bpm = _path(bpm_in, self.bpm_cnn, self.bpm_rnn)
        h_steps = _path(steps_in, self.steps_cnn, self.steps_rnn)
        return torch.cat([h_bpm, h_steps], dim=1)

    def _make_tokens(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([self.curr_proj(f) for f in feats], dim=1)

    def forward(
        self,
        bpm_in: torch.Tensor,
        steps_in: torch.Tensor,
        cur_bpm: torch.Tensor,
        cur_steps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, P = bpm_in.size(0), self.predict_windows
        past_summary = self._extract_past_summary(bpm_in, steps_in)

        feats_bpm = [
            self.agg_current_bpm(cur_bpm[:, i, :]) for i in range(P)
        ]
        feats_steps = [
            self.agg_current_steps(cur_steps[:, i, :]) for i in range(P)
        ]
        tok_bpm = self._make_tokens(feats_bpm)
        tok_steps = self._make_tokens(feats_steps)

        def _add_pos(tokens):
            seq = torch.cat([past_summary.unsqueeze(1), tokens], dim=1)
            idx = torch.arange(P + 1, device=seq.device)
            return seq + self.pos_emb(idx)[None, :, :]

        seq_bpm = _add_pos(tok_bpm)
        seq_steps = _add_pos(tok_steps)

        def _xformer(seq, attn, ln1, ffn, ln2):
            attn_out, _ = attn(seq, seq, seq)
            x = ln1(seq + attn_out)
            ffn_out = ffn(x)
            return ln2(x + ffn_out)

        res_bpm = _xformer(
            seq_bpm, self.attn_bpm, self.ln1_bpm, self.ffn_bpm, self.ln2_bpm
        )
        res_steps = _xformer(
            seq_steps, self.attn_steps, self.ln1_steps, self.ffn_steps, self.ln2_steps
        )

        out_bpm = self.fusion_bpm(res_bpm[:, 1:, :])
        out_steps = self.fusion_steps(res_steps[:, 1:, :])
        return out_bpm, out_steps

class PersonalizedForecastingModel(SSLForecastingModel):
    pass

class DrugClassifier(nn.Module):
    """Binary classifier (use / crave) built atop the CNN+GRU encoder with attention pooling."""
    def __init__(self, *, window_size: int, cfg_model: dict):
        super().__init__()
        dropout = cfg_model.get("dropout", 0.3)
        cnn_ch = cfg_model.get("cnn_channels", [32, 64])
        rnn_hidden = cfg_model.get("rnn_hidden", 128)
        rnn_layers = cfg_model.get("rnn_layers", 2)

        self.bpm_cnn = _make_cnn(cnn_ch, dropout)
        self.steps_cnn = _make_cnn(cnn_ch, dropout)
        self.bpm_rnn = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)
        self.steps_rnn = nn.GRU(cnn_ch[-1], rnn_hidden, rnn_layers, batch_first=True)

        # attention pooling: scores per timestep
        self.attn_pool = nn.Sequential(
            nn.Linear(2 * rnn_hidden, rnn_hidden),
            nn.Tanh(),
            nn.Linear(rnn_hidden, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * rnn_hidden, rnn_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(rnn_hidden, 1)
        )

    def forward(self, bpm_in: torch.Tensor, steps_in: torch.Tensor) -> torch.Tensor:
        B = bpm_in.size(0)
        # reshape to [B,1,W]
        v_bpm = bpm_in.view(B, 1, -1)
        v_steps = steps_in.view(B, 1, -1)

        # CNN + GRU per modality → [B, W, hidden]
        h_bpm, _ = self.bpm_rnn(self.bpm_cnn(v_bpm).permute(0, 2, 1))
        h_steps, _ = self.steps_rnn(self.steps_cnn(v_steps).permute(0, 2, 1))

        # merge sequences across modalities → [B, W, 2*hidden]
        h_seq = torch.cat([h_bpm, h_steps], dim=2)

        # attention pooling
        scores = self.attn_pool(h_seq)             # [B, W, 1]
        weights = torch.softmax(scores, dim=1)      # [B, W, 1]
        agg = (h_seq * weights).sum(dim=1)         # [B, 2*hidden]

        # final classifier
        return self.classifier(agg).squeeze(-1)

def partially_unfreeze_backbone(model: nn.Module, unfreeze_ratio: float = 0.3) -> None:
    params = list(model.named_parameters())
    cutoff = int(len(params) * (1.0 - unfreeze_ratio))
    for idx, (_, p) in enumerate(params):
        p.requires_grad = idx >= cutoff