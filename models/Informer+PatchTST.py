import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Informer import Model as Informer
from models.PatchTST import Model as PatchTST


class Model(nn.Module):
    """
    Informer + PatchTST 集成（残差纠错方案）

    核心设计：
    1. Informer 作为主预测器，负责大部分预测任务
    2. PatchTST 预测残差（纠错），而非重复预测
    3. 通过可学习的门控机制动态控制纠错幅度
    4. 融合权重通过 softmax 约束为 1，避免缩放退化
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len

        self.informer = Informer(configs)
        self.patchtst = PatchTST(configs)

        self.informer_scale = nn.Parameter(torch.ones(1))
        self.patchtst_scale = nn.Parameter(torch.ones(1) * 0.01)

        self.gate = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len),
            nn.Sigmoid()
        )

        self.fusion_norm = nn.LayerNorm(self.enc_in)

        self.residual_proj = nn.Sequential(
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=3, padding=1, groups=self.enc_in),
            nn.GELU(),
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=1)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        out_informer = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        out_patchtst = self.patchtst(x_enc, x_mark_enc, x_dec, x_mark_dec)

        scaled_informer = self.informer_scale * out_informer
        scaled_patchtst = self.patchtst_scale * out_patchtst

        gate_input = x_enc[:, :, 0]
        g = self.gate(gate_input).unsqueeze(-1)

        dec_out = scaled_informer + g * scaled_patchtst

        dec_out = dec_out.transpose(1, 2)
        residual_corr = self.residual_proj(dec_out)
        dec_out = dec_out + residual_corr
        dec_out = dec_out.transpose(1, 2)

        dec_out = self.fusion_norm(dec_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None
