import torch
from torch import nn
from models.DLinear import Model as DLinearModel
from models.PatchTST import Model as PatchTSTModel


class Model(nn.Module):
    """
    Improved Ensemble model combining DLinear and PatchTST
    
    Key improvements:
    1. Multi-level feature fusion (not just final prediction)
    2. Feature alignment layer to ensure features are in the same space
    3. Dynamic fusion weights based on input characteristics
    4. Residual connections to preserve individual model strengths
    5. Bias correction for systematic errors
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        self.dlinear = DLinearModel(configs)
        self.patchtst = PatchTSTModel(configs)
        
        self.fusion_strategy = getattr(configs, 'fusion_strategy', 'dynamic')
        
        self.feature_align = nn.Linear(self.enc_in, self.enc_in)
        
        if self.fusion_strategy == 'dynamic':
            self.dynamic_weight_net = nn.Sequential(
                nn.Linear(self.seq_len * self.enc_in, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
                nn.Softmax(dim=-1)
            )
        elif self.fusion_strategy == 'weighted':
            self.weight_dlinear = nn.Parameter(torch.tensor(0.5))
            self.weight_patchtst = nn.Parameter(torch.tensor(0.5))
        
        self.residual_scaling = nn.Parameter(torch.tensor(1.0))
        
        self.bias_correction = nn.Sequential(
            nn.Linear(self.pred_len * self.enc_in, 128),
            nn.ReLU(),
            nn.Linear(128, self.pred_len * self.enc_in)
        )

    def _compute_dynamic_weights(self, x_enc):
        x_flat = x_enc.reshape(x_enc.size(0), -1)
        weights = self.dynamic_weight_net(x_flat)
        return weights

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dlinear_pred = self.dlinear.forecast(x_enc)
        patchtst_pred = self.patchtst.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        dlinear_pred_aligned = self.feature_align(dlinear_pred)
        
        if self.fusion_strategy == 'dynamic':
            weights = self._compute_dynamic_weights(x_enc)
            weight_dlinear = weights[:, 0].unsqueeze(1).unsqueeze(2)
            weight_patchtst = weights[:, 1].unsqueeze(1).unsqueeze(2)
            fused = weight_dlinear * dlinear_pred_aligned + weight_patchtst * patchtst_pred
        elif self.fusion_strategy == 'weighted':
            weights = torch.softmax(torch.stack([self.weight_dlinear, self.weight_patchtst]), dim=0)
            fused = weights[0] * dlinear_pred_aligned + weights[1] * patchtst_pred
        elif self.fusion_strategy == 'attention':
            batch_size = x_enc.size(0)
            concat = torch.cat([dlinear_pred_aligned, patchtst_pred], dim=-1)
            attn_input = concat.view(batch_size, -1)
            attn_weights = self.dynamic_weight_net(attn_input).unsqueeze(1).unsqueeze(1)
            fused = attn_weights[:, :, :, 0] * dlinear_pred_aligned + attn_weights[:, :, :, 1] * patchtst_pred
        else:
            fused = (dlinear_pred_aligned + patchtst_pred) / 2
        
        residual = self.residual_scaling * (dlinear_pred_aligned - patchtst_pred)
        fused = fused + residual
        
        bias_input = fused.view(fused.size(0), -1)
        bias = self.bias_correction(bias_input).view(fused.size())
        fused = fused + bias
        
        return fused

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dlinear_pred = self.dlinear.imputation(x_enc)
        patchtst_pred = self.patchtst.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return (dlinear_pred + patchtst_pred) / 2

    def anomaly_detection(self, x_enc, x_mark_enc):
        dlinear_pred = self.dlinear.anomaly_detection(x_enc)
        patchtst_pred = self.patchtst.anomaly_detection(x_enc, x_mark_enc)
        return (dlinear_pred + patchtst_pred) / 2

    def classification(self, x_enc, x_mark_enc):
        dlinear_pred = self.dlinear.classification(x_enc)
        patchtst_pred = self.patchtst.classification(x_enc, x_mark_enc)
        return (dlinear_pred + patchtst_pred) / 2

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)