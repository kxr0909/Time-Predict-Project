import torch
import torch.nn as nn
from models.DLinear import Model as DLinearModel
from models.PatchTST import Model as PatchTSTModel
from models.Informer import Model as InformerModel


class Model(nn.Module):
    """
    三级残差融合：DLinear → PatchTST → Informer
    stage 1: 只训练 DLinear (主趋势)
    stage 2: 冻结 DLinear，训练 PatchTST (局部残差修正)
    stage 3: 冻结 DLinear+PatchTST，训练 Informer (全局长程残差修正)
    stage 4: 全解冻微调
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.dlinear = DLinearModel(configs)
        self.patchtst = PatchTSTModel(configs)
        self.informer = InformerModel(configs)

        self.patchtst_scale = nn.Parameter(torch.ones(1, 1, self.enc_in) * 0.01)
        self.informer_scale = nn.Parameter(torch.ones(1, 1, self.enc_in) * 0.001)

        self.bias_correction = nn.Sequential(
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=1, groups=self.enc_in),
            nn.GELU(),
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=1, groups=self.enc_in)
        )

        self.fusion_norm = nn.LayerNorm(self.enc_in)

    def set_stage(self, stage=1):
        if stage == 1:
            for p in self.dlinear.parameters(): p.requires_grad = True
            for p in self.patchtst.parameters(): p.requires_grad = False
            for p in self.informer.parameters(): p.requires_grad = False
            for p in self.bias_correction.parameters(): p.requires_grad = False
            self.patchtst_scale.requires_grad = False
            self.informer_scale.requires_grad = False
            self.patchtst_scale.data.fill_(0.01)
            self.informer_scale.data.fill_(0.001)
            print(">>> [Stage 1] 只训练 DLinear")
        elif stage == 2:
            for p in self.dlinear.parameters(): p.requires_grad = False
            for p in self.patchtst.parameters(): p.requires_grad = True
            for p in self.informer.parameters(): p.requires_grad = False
            for p in self.bias_correction.parameters(): p.requires_grad = True
            self.patchtst_scale.requires_grad = True
            self.informer_scale.requires_grad = False
            self.patchtst_scale.data.fill_(0.01)
            self.informer_scale.data.fill_(0.001)
            print(">>> [Stage 2] 冻结 DLinear，训练 PatchTST")
        elif stage == 3:
            for p in self.dlinear.parameters(): p.requires_grad = False
            for p in self.patchtst.parameters(): p.requires_grad = False
            for p in self.informer.parameters(): p.requires_grad = True
            for p in self.bias_correction.parameters(): p.requires_grad = True
            self.patchtst_scale.requires_grad = True
            self.informer_scale.requires_grad = True
            self.informer_scale.data.fill_(0.001)
            print(">>> [Stage 3] 冻结 DLinear+PatchTST，训练 Informer")
        elif stage == 4:
            for p in self.parameters(): p.requires_grad = True
            print(">>> [Stage 4] 全部解冻，全模型微调")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dlinear_pred = self.dlinear.forecast(x_enc)
        patchtst_pred = self.patchtst.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        informer_pred = self.informer(x_enc, x_mark_enc, x_dec, x_mark_dec)

        pred = dlinear_pred + self.patchtst_scale * patchtst_pred + self.informer_scale * informer_pred

        bias_input = pred.transpose(1, 2)
        bias = self.bias_correction(bias_input).transpose(1, 2)
        pred = pred + bias

        pred = self.fusion_norm(pred)
        return pred

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None
