# src/models.py
import torch
import torch.nn as nn

class SSLForecastingModel(nn.Module):
    """
    Self‑supervised multi‑step forecaster with *cross‑modal* current‑window
    tokens (no self‑leakage):
        • Path‑BPM  : past‑summary  + current‑STEPS  tokens  → predict BPM
        • Path‑Steps: past‑summary  + current‑BPM    tokens  → predict Steps
    """
    def __init__(self,
                 window_size: int,
                 predict_windows: int,
                 attn_heads: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.window_size     = window_size
        self.predict_windows = predict_windows
        self.embed_dim       = 256

        # ── per‑modality CNN+GRU encoders for the *past* windows ──────────
        def make_branch():
            return nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.bpm_cnn   = make_branch()
        self.steps_cnn = make_branch()
        self.bpm_rnn   = nn.GRU(64, 128, num_layers=2, batch_first=True)
        self.steps_rnn = nn.GRU(64, 128, num_layers=2, batch_first=True)

        # ── single‑modality aggregators for the *current* window ──────────
        self.agg_current_bpm   = nn.Sequential(
            nn.Linear(window_size, 16), nn.ReLU(), nn.Dropout(dropout)
        )
        self.agg_current_steps = nn.Sequential(
            nn.Linear(window_size, 16), nn.ReLU(), nn.Dropout(dropout)
        )
        # shared projection 16 → 256
        self.curr_proj = nn.Linear(16, self.embed_dim)

        # positional embeddings (P+1 because we prepend the past summary)
        self.pos_emb = nn.Embedding(predict_windows + 1, self.embed_dim)

        # two separate attention‑>FFN stacks (one per prediction path)
        def make_attn_block():
            attn = nn.MultiheadAttention(self.embed_dim, attn_heads,
                                         batch_first=True)
            ln1  = nn.LayerNorm(self.embed_dim)
            ffn  = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Dropout(dropout),
            )
            ln2  = nn.LayerNorm(self.embed_dim)
            return attn, ln1, ffn, ln2

        (self.attn_bpm,
         self.ln1_bpm,
         self.ffn_bpm,
         self.ln2_bpm)   = make_attn_block()

        (self.attn_steps,
         self.ln1_steps,
         self.ffn_steps,
         self.ln2_steps) = make_attn_block()

        # modality‑specific decoders
        self.fusion_bpm   = nn.Linear(self.embed_dim, window_size)
        self.fusion_steps = nn.Linear(self.embed_dim, window_size)

    # ──────────────────────────────────────────────────────────────────────
    def _extract_past_summary(self, bpm_in, steps_in):
        """CNN+GRU → last hidden → concatenate → 256‑d past summary"""
        B = bpm_in.size(0)

        def feat(x, cnn, rnn):
            v = x.view(B, -1).unsqueeze(1)        # [B,1,L]
            v = cnn(v).permute(0, 2, 1)           # [B,L,64]
            h, _ = rnn(v)
            return h[:, -1, :]                    # [B,128]

        h_bpm   = feat(bpm_in,   self.bpm_cnn,   self.bpm_rnn)
        h_steps = feat(steps_in, self.steps_cnn, self.steps_rnn)
        return torch.cat([h_bpm, h_steps], dim=1)  # [B,256]

    def _build_tokens(self, features, P):
        """features: list length P of 16‑d tensors → [B,P,256]"""
        tokens = [self.curr_proj(f) for f in features]         # each [B,256]
        return torch.stack(tokens, dim=1)                      # [B,P,256]

    # ------------------------------------------------------------------ #
    def forward(self, bpm_in, steps_in, cur_bpm, cur_steps):
        """
        Shapes
        -------
        bpm_in, steps_in : [B, input_windows, W]
        cur_bpm,cur_steps: [B, P, W]  (P = predict_windows, W = window_size)
        """
        B, P, W = bpm_in.size(0), self.predict_windows, self.window_size

        past_summary = self._extract_past_summary(bpm_in, steps_in)  # [B,256]

        # ── build cross‑modal token sequences ───────────────────────────
        # Path‑BPM  sees current‑steps only
        feats_bpm = [self.agg_current_steps(cur_steps[:, w, :]) for w in range(P)]
        tok_bpm   = self._build_tokens(feats_bpm, P)                # [B,P,256]

        # Path‑Steps sees current‑bpm only
        feats_steps = [self.agg_current_bpm(cur_bpm[:, w, :]) for w in range(P)]
        tok_steps   = self._build_tokens(feats_steps, P)            # [B,P,256]

        # prepend past summary & add positional enc.
        def add_pos(tokens):
            seq = torch.cat([past_summary.unsqueeze(1), tokens], dim=1)  # [B,P+1,256]
            idx = torch.arange(P + 1, device=seq.device)
            return seq + self.pos_emb(idx)[None, :, :]
        seq_bpm   = add_pos(tok_bpm)
        seq_steps = add_pos(tok_steps)

        # ── transformer block (per path) ────────────────────────────────
        # BPM path
        attn_out, _ = self.attn_bpm(seq_bpm, seq_bpm, seq_bpm)
        res1 = self.ln1_bpm(seq_bpm + attn_out)
        ffn_out = self.ffn_bpm(res1)
        res_bpm = self.ln2_bpm(res1 + ffn_out)

        # Steps path
        attn_out_s, _ = self.attn_steps(seq_steps, seq_steps, seq_steps)
        res1_s = self.ln1_steps(seq_steps + attn_out_s)
        ffn_out_s = self.ffn_steps(res1_s)
        res_steps = self.ln2_steps(res1_s + ffn_out_s)

        # ── decode future tokens (drop index 0 = past summary) ──────────
        out_bpm   = self.fusion_bpm(res_bpm[:, 1:, :])    # [B,P,W]
        out_steps = self.fusion_steps(res_steps[:, 1:, :])# [B,P,W]
        return out_bpm, out_steps


class PersonalizedForecastingModel(SSLForecastingModel):
    """Same architecture as SSLForecastingModel; used for fine‑tuning."""
    pass


class DrugClassifier(nn.Module):
    """
    Binary classifier that re‑uses the CNN+GRU feature extractors
    (kept identical), followed by a small MLP head.
    """
    def __init__(self, window_size=6, dropout=0.3):
        super().__init__()
        self.window_size = window_size

        def make_branch():
            return nn.Sequential(
                nn.Conv1d(1, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.bpm_cnn   = make_branch()
        self.steps_cnn = make_branch()
        self.bpm_rnn   = nn.GRU(64, 128, num_layers=2, batch_first=True)
        self.steps_rnn = nn.GRU(64, 128, num_layers=2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, bpm_in, steps_in):
        B = bpm_in.size(0)

        def feat(x, cnn, rnn):
            v = x.view(B, -1).unsqueeze(1)
            v = cnn(v).permute(0, 2, 1)
            h, _ = rnn(v)
            return h[:, -1, :]

        z_bpm   = feat(bpm_in,   self.bpm_cnn,   self.bpm_rnn)
        z_steps = feat(steps_in, self.steps_cnn, self.steps_rnn)
        fused   = torch.cat([z_bpm, z_steps], dim=1)  # [B,256]
        return self.classifier(fused).squeeze(-1)


def partially_unfreeze_backbone(model, unfreeze_ratio=0.3):
    params = list(model.named_parameters())
    cutoff = int(len(params) * (1 - unfreeze_ratio))
    for i, (_, p) in enumerate(params):
        p.requires_grad = (i >= cutoff)
