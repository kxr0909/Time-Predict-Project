import torch
import torch.nn as nn
from models.DLinear import Model as DLinearModel
from models.PatchTST import Model as PatchTSTModel

class Model(nn.Module):
    """
    级联残差融合方案（优化版）：
    DLinear 预测主趋势与周期，PatchTST 预测全局残差细节。
    保持 Channel Independence (通道独立性)，避免维度崩溃。
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 第一级：DLinear 负责趋势+周期主预测
        self.dlinear = DLinearModel(configs)
        # 第二级：PatchTST 负责拟合残差，修正细节
        self.patchtst = PatchTSTModel(configs)
        
        # 将残差缩放系数改为通道独立的 Parameter，让不同变量有不同的缩放力度
        self.residual_scale = nn.Parameter(torch.ones(1, 1, self.enc_in) * 0.5)
        
        # 改进的偏置校正：使用 1D 分组卷积保持通道独立性 (Groups = enc_in)
        # 仅在时间维度上做微小的线性微调，避免过拟合与跨通道干扰
        self.bias_correction = nn.Sequential(
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=1, groups=self.enc_in),
            nn.GELU(),
            nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in, kernel_size=1, groups=self.enc_in)
        )

    def set_stage(self, stage=1):
        """
        核心控制函数：通过冻结/放开参数，解决“梯度打架”和“残差淹没”
        stage=1: 仅训练 DLinear（抓取主趋势）
        stage=2: 冻结 DLinear，专心训练 PatchTST（抓取残差）
        stage=3: 全解冻微调
        """
        if stage == 1:
            for p in self.dlinear.parameters(): p.requires_grad = True
            for p in self.patchtst.parameters(): p.requires_grad = False
            for p in self.bias_correction.parameters(): p.requires_grad = False
            self.residual_scale.requires_grad = False
            print(">>> [Stage 1] 只有 DLinear 参与训练，锁定 PatchTST。")
        elif stage == 2:
            for p in self.dlinear.parameters(): p.requires_grad = False
            for p in self.patchtst.parameters(): p.requires_grad = True
            for p in self.bias_correction.parameters(): p.requires_grad = True
            self.residual_scale.requires_grad = True
            print(">>> [Stage 2] 锁定 DLinear，专门训练 PatchTST 拟合残差。")
        elif stage == 3:
            for p in self.parameters(): p.requires_grad = True
            print(">>> [Stage 3] 全部解冻，全模型微调融合。")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. DLinear 输出主预测 (Shape: [B, pred_len, enc_in])
        dlinear_pred = self.dlinear.forecast(x_enc)
        
        # 2. PatchTST 接收完整上下文并输出预测
        patchtst_pred = self.patchtst.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # 3. 趋势 + 缩放后的残差融合
        final_pred = dlinear_pred + self.residual_scale * patchtst_pred
        
        # 4. 轻量通道独立偏置校正
        bias_input = final_pred.transpose(1, 2) 
        bias = self.bias_correction(bias_input).transpose(1, 2)
        
        final_pred = final_pred + bias
        return final_pred

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
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc, x_mark_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        else:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)