# src/models.py

import torch
import torch.nn as nn

class SSLForecastingModel(nn.Module):
    """
    A multi-step forecasting model with self-attention over past + future windows.
    Current‑windows (BPM/Steps) are concatenated, not averaged.
    """
    def __init__(self,
                 window_size: int,
                 predict_windows: int,
                 attn_heads: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.window_size     = window_size
        self.predict_windows = predict_windows

        # --- BPM branch ---
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bpm_rnn = nn.GRU(64, 128, num_layers=2, batch_first=True)

        # --- Steps branch ---
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.steps_rnn = nn.GRU(64, 128, num_layers=2, batch_first=True)

        # current window embeddings → 16d each
        self.agg_current_bpm   = nn.Sequential(
            nn.Linear(window_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.agg_current_steps = nn.Sequential(
            nn.Linear(window_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # concatenate (16+16=32) → project to 256d tokens
        self.curr_proj = nn.Linear(16*2, 256)

        # positional embeddings for [past + predict_windows]
        self.pos_emb = nn.Embedding(predict_windows + 1, 256)

        # self-attention over tokens
        self.attn    = nn.MultiheadAttention(embed_dim=256,
                                             num_heads=attn_heads,
                                             batch_first=True)
        self.attn_ln = nn.LayerNorm(256)

        # feed-forward residual block
        self.ffn    = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(256)

        # final fusion heads
        self.fusion_bpm   = nn.Linear(256, window_size)
        self.fusion_steps = nn.Linear(256, window_size)

    def forward(self, bpm_in, steps_in, cur_bpm, cur_steps):
        B, P = bpm_in.size(0), self.predict_windows

        # extract past summary
        def extract(x, cnn, rnn):
            v = x.view(B, -1).unsqueeze(1)   # [B,1,past_len]
            v = cnn(v).permute(0,2,1)        # [B,past_len,64]
            h, _ = rnn(v)                    # [B,past_len,128]
            return h[:, -1, :]               # [B,128]

        h_bpm   = extract(bpm_in,   self.bpm_cnn,   self.bpm_rnn)
        h_steps = extract(steps_in, self.steps_cnn, self.steps_rnn)
        past    = torch.cat([h_bpm, h_steps], dim=1)  # [B,256]

        # build token sequence: [past, cw1, cw2, …]
        tokens = [past]
        for w in range(P):
            cb = self.agg_current_bpm(cur_bpm[:,w,:])    # [B,16]
            cs = self.agg_current_steps(cur_steps[:,w,:])# [B,16]
            cw = torch.cat([cb, cs], dim=1)              # [B,32]
            tokens.append(self.curr_proj(cw))            # [B,256]
        tokens = torch.stack(tokens, dim=1)              # [B,P+1,256]

        # add positional embeddings
        idx = torch.arange(P+1, device=tokens.device)
        tokens = tokens + self.pos_emb(idx)[None,:,:]

        # attention + residual + norm
        attn_out, _ = self.attn(tokens, tokens, tokens)
        res1 = self.attn_ln(tokens + attn_out)

        # feed-forward + residual + norm
        ffn_out = self.ffn(res1)
        res2    = self.ffn_ln(res1 + ffn_out)

        # decode each future token
        fused     = res2[:,1:,:]               # [B,P,256]
        out_bpm   = self.fusion_bpm(fused)     # [B,P,window_size]
        out_steps = self.fusion_steps(fused)   # [B,P,window_size]
        return out_bpm, out_steps


class PersonalizedForecastingModel(SSLForecastingModel):
    """Same as SSLForecastingModel; used for fine‑tuning."""
    pass


class DrugClassifier(nn.Module):
    """
    Binary classifier re‑using the same CNN+RNN extractors.
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
        self.bpm_rnn   = nn.GRU(64, 128, num_layers=2, batch_first=True)
        self.steps_cnn = make_branch()
        self.steps_rnn = nn.GRU(64, 128, num_layers=2, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, bpm_in, steps_in):
        B = bpm_in.size(0)
        def feat(x, cnn, rnn):
            v = x.view(B, -1).unsqueeze(1)
            v = cnn(v).permute(0,2,1)
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
