# models.py

import torch
import torch.nn as nn

class SSLForecastingModel(nn.Module):
    """
    A multi-step forecasting model used for self-supervised pretraining.
    Same basic architecture as your original pretrain scripts, but
    adapted to the new window sizes if desired.
    """
    def __init__(self, window_size=6):
        super(SSLForecastingModel, self).__init__()
        self.window_size = window_size

        # CNN + LSTM for BPM
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.bpm_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

        # CNN + LSTM for Steps
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.steps_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

        # Aggregators
        self.agg_current_bpm = nn.Sequential(
            nn.Linear(window_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.agg_current_steps = nn.Sequential(
            nn.Linear(window_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion heads
        self.fusion_bpm = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, window_size)
        )
        self.fusion_steps = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, window_size)
        )

    def forward(self, bpm_input, steps_input, curr_bpm_windows, curr_steps_windows):
        B = bpm_input.size(0)

        # Flatten for CNN
        bpm_seq = bpm_input.view(B, -1).unsqueeze(1)
        bpm_cnn_out = self.bpm_cnn(bpm_seq)
        bpm_cnn_out = bpm_cnn_out.permute(0, 2, 1)
        bpm_lstm_out, _ = self.bpm_lstm(bpm_cnn_out)
        bpm_hidden = bpm_lstm_out[:, -1, :]  # [B,128]

        steps_seq = steps_input.view(B, -1).unsqueeze(1)
        steps_cnn_out = self.steps_cnn(steps_seq)
        steps_cnn_out = steps_cnn_out.permute(0, 2, 1)
        steps_lstm_out, _ = self.steps_lstm(steps_cnn_out)
        steps_hidden = steps_lstm_out[:, -1, :]  # [B,128]

        past_features = torch.cat([bpm_hidden, steps_hidden], dim=1)  # [B,256]

        bpm_out_list = []
        steps_out_list = []
        pred_w = curr_bpm_windows.size(1)  # PREDICT_WINDOWS

        for w_idx in range(pred_w):
            # Gather "current" input for aggregator
            curr_bpm_1win = curr_bpm_windows[:, w_idx, :]   # shape [B, window_size]
            curr_steps_1win = curr_steps_windows[:, w_idx, :]

            # embed them
            curr_bpm_emb = self.agg_current_bpm(curr_bpm_1win)
            curr_steps_emb = self.agg_current_steps(curr_steps_1win)

            # BPM => fuse with steps embedding
            bpm_fusion_in = torch.cat([past_features, curr_steps_emb], dim=1)
            bpm_pred_1win = self.fusion_bpm(bpm_fusion_in)  # => [B,window_size]

            # Steps => fuse with BPM embedding
            steps_fusion_in = torch.cat([past_features, curr_bpm_emb], dim=1)
            steps_pred_1win = self.fusion_steps(steps_fusion_in)

            bpm_out_list.append(bpm_pred_1win.unsqueeze(1))
            steps_out_list.append(steps_pred_1win.unsqueeze(1))

        bpm_pred_final = torch.cat(bpm_out_list, dim=1)      # [B,PREDICT_WINDOWS,window_size]
        steps_pred_final = torch.cat(steps_out_list, dim=1)  # same shape
        return bpm_pred_final, steps_pred_final


class PersonalizedForecastingModel(nn.Module):
    """
    We can re-use the same architecture for personalized fine-tuning, or adapt it slightly.
    Here, we keep the same structure for simplicity; we just load the SSL weights
    and partially unfreeze the layers.
    """
    def __init__(self, window_size=6):
        super(PersonalizedForecastingModel, self).__init__()
        self.window_size = window_size

        # replicate the same architecture
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.bpm_lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)

        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.steps_lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)

        self.agg_current_bpm = nn.Sequential(
            nn.Linear(window_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.agg_current_steps = nn.Sequential(
            nn.Linear(window_size, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fusion_bpm = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, window_size)
        )
        self.fusion_steps = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, window_size)
        )

    def forward(self, bpm_input, steps_input, curr_bpm, curr_steps):
        B = bpm_input.size(0)
        # same logic
        bpm_seq = bpm_input.view(B, -1).unsqueeze(1)
        bpm_cnn_out = self.bpm_cnn(bpm_seq)
        bpm_cnn_out = bpm_cnn_out.permute(0,2,1)
        bpm_lstm_out, _ = self.bpm_lstm(bpm_cnn_out)
        bpm_hidden = bpm_lstm_out[:, -1, :]

        steps_seq = steps_input.view(B, -1).unsqueeze(1)
        steps_cnn_out = self.steps_cnn(steps_seq)
        steps_cnn_out = steps_cnn_out.permute(0,2,1)
        steps_lstm_out, _ = self.steps_lstm(steps_cnn_out)
        steps_hidden = steps_lstm_out[:,-1,:]

        past_features = torch.cat([bpm_hidden, steps_hidden], dim=1)

        out_bpm = []
        out_steps = []
        pred_w = curr_bpm.size(1)

        for w in range(pred_w):
            c_bpm_1w = curr_bpm[:, w, :]
            c_steps_1w = curr_steps[:, w, :]

            cbpm_emb = self.agg_current_bpm(c_bpm_1w)
            cstep_emb = self.agg_current_steps(c_steps_1w)

            bpm_fusion_in = torch.cat([past_features, cstep_emb], dim=1)
            steps_fusion_in = torch.cat([past_features, cbpm_emb], dim=1)

            pred_bpm_1win = self.fusion_bpm(bpm_fusion_in)
            pred_steps_1win = self.fusion_steps(steps_fusion_in)
            out_bpm.append(pred_bpm_1win.unsqueeze(1))
            out_steps.append(pred_steps_1win.unsqueeze(1))

        final_bpm = torch.cat(out_bpm, dim=1)
        final_steps = torch.cat(out_steps, dim=1)
        return final_bpm, final_steps


class DrugClassifier(nn.Module):
    """
    Final classification model, which uses the personalized backbone’s “feature extraction”
    layers for BPM & Steps, then a classification head for drug use detection.
    We'll define a simpler forward pass that just extracts a [B,256] feature vector from the partial
    CNN+LSTM, then do a final Dense => sigmoid for 0/1 classification.
    """
    def __init__(self, window_size=6):
        super(DrugClassifier, self).__init__()
        self.window_size = window_size

        # same CNN+LSTM but we won't use the 'fusion' part
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.bpm_lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)

        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.steps_lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # for BCEWithLogits
        )

    def forward(self, bpm_in, steps_in):
        B = bpm_in.size(0)

        bpm_seq = bpm_in.view(B, -1).unsqueeze(1)
        bpm_cnn_out = self.bpm_cnn(bpm_seq)
        bpm_cnn_out = bpm_cnn_out.permute(0,2,1)
        bpm_lstm_out, _ = self.bpm_lstm(bpm_cnn_out)
        bpm_hidden = bpm_lstm_out[:, -1, :]

        steps_seq = steps_in.view(B, -1).unsqueeze(1)
        steps_cnn_out = self.steps_cnn(steps_seq)
        steps_cnn_out = steps_cnn_out.permute(0,2,1)
        steps_lstm_out, _ = self.steps_lstm(steps_cnn_out)
        steps_hidden = steps_lstm_out[:,-1,:]

        fused = torch.cat([bpm_hidden, steps_hidden], dim=1)  # [B,256]
        logits = self.classifier(fused)
        return logits.squeeze(-1)


def partially_unfreeze_backbone(model, unfreeze_ratio=0.3):
    """
    Example: unfreeze only some fraction of parameters in the CNN+LSTM.
    unfreeze_ratio=0.3 => unfreeze 30% from last to first.
    Adjust logic to your preference.
    """
    # Flatten all the named parameters in a list
    all_params = []
    for name, param in model.named_parameters():
        all_params.append((name, param))

    # We'll sort them so that we unfreeze "later" layers first
    # The logic here is simplistic. You can define your own strategy.
    # For demonstration, we just unfreeze the last 30% of param list.
    total_count = len(all_params)
    boundary = int(total_count * (1 - unfreeze_ratio))

    for i, (name, param) in enumerate(all_params):
        if i >= boundary:
            param.requires_grad = True
        else:
            param.requires_grad = False
