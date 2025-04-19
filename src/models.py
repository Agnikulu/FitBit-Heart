import torch
import torch.nn as nn

# ------------------------------------------------------------------ #
# 1. SSL backbone (unchanged)                                        #
# ------------------------------------------------------------------ #
class SSLForecastingModel(nn.Module):
    """
    Multi‑step forecasting backbone used for self‑supervised pre‑training.
    """
    def __init__(self, window_size=6):
        super().__init__()
        self.window_size = window_size
        # ------- BPM branch -------
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.3)
        )
        self.bpm_lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
        # ------- Steps branch -----
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.3)
        )
        self.steps_lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
        # ------- current window embeddings -------
        self.agg_current_bpm   = nn.Sequential(nn.Linear(window_size,16), nn.ReLU(), nn.Dropout(0.3))
        self.agg_current_steps = nn.Sequential(nn.Linear(window_size,16), nn.ReLU(), nn.Dropout(0.3))
        # ------- fusion heads -------
        self.fusion_bpm = nn.Sequential(
            nn.Linear(256+16,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,window_size))
        self.fusion_steps = nn.Sequential(
            nn.Linear(256+16,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,window_size))

    def forward(self, bpm_in, steps_in, cur_bpm, cur_steps):
        B = bpm_in.size(0)
        def branch(x, cnn, lstm):
            x = x.view(B,-1).unsqueeze(1)   # [B,1,T]
            x = cnn(x).permute(0,2,1)       # [B,T,C]
            return lstm(x)[0][:,-1,:]       # last hidden
        h_bpm   = branch(bpm_in,   self.bpm_cnn,   self.bpm_lstm)
        h_steps = branch(steps_in, self.steps_cnn, self.steps_lstm)
        past = torch.cat([h_bpm,h_steps], dim=1)    # [B,256]

        out_bpm, out_steps = [], []
        P = cur_bpm.size(1)
        for w in range(P):
            cb, cs = cur_bpm[:,w,:], cur_steps[:,w,:]
            emb_cb, emb_cs = self.agg_current_bpm(cb), self.agg_current_steps(cs)
            out_bpm.append(self.fusion_bpm(torch.cat([past,emb_cs],dim=1)).unsqueeze(1))
            out_steps.append(self.fusion_steps(torch.cat([past,emb_cb],dim=1)).unsqueeze(1))
        return torch.cat(out_bpm,1), torch.cat(out_steps,1)

# ------------------------------------------------------------------ #
# 2. Personalized fine‑tune model (architecture identical)           #
# ------------------------------------------------------------------ #
class PersonalizedForecastingModel(SSLForecastingModel):
    """Same layers; we inherit forward from parent."""
    pass

# ------------------------------------------------------------------ #
# 3. Binary drug/crave classifier                                    #
# ------------------------------------------------------------------ #
class DrugClassifier(nn.Module):
    """
    Re‑uses BPM & Steps feature extractors; adds 2‑layer classifier.
    """
    def __init__(self, window_size=6):
        super().__init__()
        self.window_size = window_size
        # feature branches (same as backbone)
        self.bpm_cnn   = nn.Sequential(
            nn.Conv1d(1,32,3,padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(), nn.Dropout(0.3))
        self.bpm_lstm  = nn.LSTM(64,128,2,batch_first=True)
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1,32,3,padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(), nn.Dropout(0.3))
        self.steps_lstm= nn.LSTM(64,128,2,batch_first=True)
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,1))

    def forward(self, bpm_in, steps_in):
        B = bpm_in.size(0)
        def feat(x, cnn, lstm):
            x = x.view(B,-1).unsqueeze(1)
            x = cnn(x).permute(0,2,1)
            return lstm(x)[0][:,-1,:]
        f_bpm   = feat(bpm_in, self.bpm_cnn, self.bpm_lstm)
        f_steps = feat(steps_in, self.steps_cnn, self.steps_lstm)
        return self.classifier(torch.cat([f_bpm,f_steps],1)).squeeze(-1)

# ------------------------------------------------------------------ #
# 4. fine‑tune helper                                                #
# ------------------------------------------------------------------ #
def partially_unfreeze_backbone(model, unfreeze_ratio=0.3):
    params = [(n,p) for n,p in model.named_parameters()]
    k = int(len(params)*(1-unfreeze_ratio))
    for i,(n,p) in enumerate(params):
        p.requires_grad = i >= k
