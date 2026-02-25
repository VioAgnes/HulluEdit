"""
ECSE 核心模块：证据子空间编辑器
实现在线子空间估计、信任域门控和证据证书
"""
from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class ECSEConfig:
    """ECSE 配置"""
    rank_evidence: int = 6      # 证据子空间秩 r
    rank_prior: int = 4          # 反先验子空间秩 q
    kappa: float = 0.6           # 信任域门控系数
    lambda_prior: float = 0.3    # 先验抑制强度
    eps: float = 1e-6            # 数值稳定性参数
    # 闭式编辑的稳健化超参
    lambda_n_max: float = 4.0     # λ_n 上限，防止过度收缩
    lambda_p_max: float = 4.0     # λ_p 上限，防止过度收缩
    vcr_floor: float = 0.05       # VCR 下限，抑制 λ_n 爆炸
    pcr_ceiling: float = 0.95     # PCR 上限，抑制 λ_p 爆炸
    pcr_threshold: float = 0.02   # 低冲突阈值，低于则不抑制先验
    blend_tau: float = 0.7        # 与原始 h 的混合比：h_out = (1-τ)h + τ h_closed
    norm_preserve: bool = True    # 是否恢复范数尺度
    norm_beta: float = 0.5        # 范数恢复指数：s^β, s=||h||/||h'||
    weight_temp: float = 1.5      # 视觉权重温度：softmax(cos/温度)
    # 消融与变体控制
    uniform_svd: bool = False     # 证据子空间提取时不使用加权（均匀SVD）
    no_complement: bool = False   # 反先验子空间不投影到视觉补空间（关闭正交性）
    no_gating: bool = False       # 关闭基于证书的门控（lambda_n 使用固定值）
    use_fixed_strengths: bool = False  # 固定编辑强度（覆盖自适应）
    fixed_lambda_n: float = 0.0   # 固定残差收缩强度
    fixed_lambda_p: float = 0.0   # 固定反先验收缩强度
    only_residual: bool = False   # 仅缩减残差分量（禁用反先验）
    only_anti_prior: bool = False # 仅缩减反先验分量（禁用残差缩减）


@dataclass
class ECSEReturn:
    """ECSE 编辑返回值"""
    h_edited: Tensor      # 编辑后的隐藏状态 [d]
    gate: Tensor          # 门控值 (scalar)
    ecr: Tensor           # 证据覆盖率 Evidence Coverage Ratio
    epc: Tensor           # 证据先验冲突 Evidence-Prior Conflict
    U: Tensor             # 证据子空间基 [d, r]
    P: Tensor             # 反先验子空间基 [d, q]


class ECSESteerer:
    """ECSE 引导器：在线子空间估计与最小范数编辑"""
    
    def __init__(self, config: ECSEConfig):
        self.cfg = config

    @staticmethod
    def _weighted_svd(X: Tensor, w: Optional[Tensor], k: int) -> Tensor:
        """
        使用加权SVD提取前k个主成分方向（适用于 n << d 的情况）
        
        当样本数远小于维度时，直接对数据矩阵做SVD，避免构造大的协方差矩阵。
        这在数值上更稳定、更高效。
        
        Args:
            X: [n, d] 样本矩阵，通常 n << d（如视觉token: 576 << 4096）
            w: [n] 权重向量（可选），用于加权PCA
            k: 主成分数量
        Returns:
            V: [d, k_actual] 前k个主成分方向（按奇异值降序排列，实际可能小于k）
        """
        if k <= 0 or X.numel() == 0:
            d = X.shape[-1] if X.numel() > 0 else 4096
            return X.new_zeros((d, 0))
        
        n, d = X.shape
        if n == 0 or d == 0:
            return X.new_zeros((d, 0))
        
        # 中心化
        if w is None:
            Xc = X - X.mean(dim=0, keepdim=True)
        else:
            # 加权中心化
            if w.shape[0] != n:
                raise ValueError(f"权重维度 {w.shape[0]} 与样本数 {n} 不匹配")
            w = w / (w.sum() + 1e-12)
            mu = (w[:, None] * X).sum(dim=0, keepdim=True)
            Xc = X - mu
        
        # 确定实际能提取的k值（不能超过rank）
        max_rank = min(n, d)
        k_actual = min(k, max_rank)
        if k_actual == 0:
            return Xc.new_zeros((d, 0))
        
        # 应用权重（加权SVD：相当于对加权后的数据做SVD）
        if w is not None:
            # 加权：X_weighted = diag(sqrt(w)) @ Xc
            # 确保权重为正且数值稳定
            w_sqrt = torch.sqrt(torch.clamp(w, min=0.0))
            Xc = Xc * w_sqrt[:, None]
        
        # 对 [n, d] 矩阵做SVD
        # Xc = U_n @ Σ @ V^T，其中：
        # - U_n: [n, min(n,d)] 左奇异向量
        # - Σ: [min(n,d)] 奇异值（降序）
        # - Vt: [min(n,d), d] 右奇异向量（转置后），Vt[i] 是第i个主成分方向
        orig_dtype = Xc.dtype
        Xc_float = Xc.float() if Xc.dtype == torch.float16 else Xc
        
        try:
            # full_matrices=False: 只计算 min(n,d) 个奇异向量，更高效
            # 检查数值稳定性：如果矩阵太小或全是零，直接返回零矩阵
            if Xc_float.abs().max() < 1e-10:
                return Xc.new_zeros((d, k_actual))
            
            _, _, Vt = torch.linalg.svd(Xc_float, full_matrices=False)
            # Vt: [min(n,d), d]，每行是一个主成分方向（按奇异值降序）
            # 确保k_actual不超过Vt的第一维（理论上k_actual <= Vt.shape[0]，但为了安全还是检查）
            k_extract = min(k_actual, Vt.shape[0])
            if k_extract > 0:
                V = Vt[:k_extract, :].T.to(orig_dtype)  # [d, k_extract]
            else:
                V = Xc.new_zeros((d, 0))
        except RuntimeError as e:
            # 如果SVD失败（理论上不应该），降级到原来的方法
            print(f"[WARN] SVD失败，降级到协方差方法: {e}")
            # 降级处理：重新计算中心化数据，使用协方差方法但需要更强的正则化
            # 注意：这里需要重新计算，因为Xc可能已经被修改（加权）
            if w is not None:
                w_norm = w / (w.sum() + 1e-12)
                mu = (w_norm[:, None] * X).sum(dim=0, keepdim=True)
                Xc_fallback = X - mu
                # 加权协方差：C = Xc^T @ diag(w) @ Xc
                C = Xc_fallback.T @ (w_norm[:, None] * Xc_fallback)
            else:
                Xc_fallback = X - X.mean(dim=0, keepdim=True)
                # 标准协方差
                C = Xc_fallback.T @ Xc_fallback / max(1, X.shape[0] - 1)
            # 使用更强的正则化
            C_reg = C.float() + 1e-3 * torch.eye(C.shape[0], device=C.device, dtype=torch.float32)
            evals, evecs = torch.linalg.eigh(C_reg)
            # 确保k不超过特征向量的数量（降级路径中的k_actual应该与正常路径一致）
            # 这里使用与正常路径相同的k_actual计算方式
            max_rank_fallback = min(n, d)  # 重新计算max_rank（n和d在函数开始已定义）
            k_actual_fallback = min(k, max_rank_fallback)
            k_extract = min(k_actual_fallback, evecs.shape[1])
            if k_extract > 0:
                V = evecs[:, -k_extract:].to(orig_dtype)
            else:
                V = Xc.new_zeros((d, 0))
        
        return V

    @staticmethod
    def _weighted_cov(X: Tensor, w: Optional[Tensor]) -> Tensor:
        """
        计算加权协方差矩阵（已弃用，保留用于向后兼容）
        Args:
            X: [n, d] 样本矩阵
            w: [n] 权重向量（可选）
        Returns:
            C: [d, d] 协方差矩阵
        """
        if w is None:
            # 无权重：标准协方差
            Xc = X - X.mean(dim=0, keepdim=True)
            return Xc.T @ Xc / max(1, X.shape[0] - 1)
        # 加权协方差
        w = w / (w.sum() + 1e-12)
        mu = (w[:, None] * X).sum(dim=0, keepdim=True)
        Xc = X - mu
        return (Xc.T * w) @ Xc

    @staticmethod
    def _top_eigvecs(C: Tensor, k: int, reg: float = 1e-5) -> Tensor:
        """
        提取协方差矩阵的前 k 个主特征向量（已弃用，保留用于向后兼容）
        Args:
            C: [d, d] 对称正定矩阵
            k: 特征向量数量
            reg: 正则化系数，用于稳定病态矩阵
        Returns:
            V: [d, k] 前 k 个特征向量（按特征值降序）
        """
        if k <= 0:
            return C.new_zeros((C.shape[0], 0))
        # linalg.eigh 在 CUDA 上不支持 Half，需要转换为 float32
        orig_dtype = C.dtype
        C_float = C.float() if C.dtype == torch.float16 else C
        
        # 添加正则化防止矩阵病态
        C_reg = C_float + reg * torch.eye(C_float.shape[0], device=C_float.device, dtype=C_float.dtype)
        
        try:
            evals, evecs = torch.linalg.eigh(C_reg)  # 升序排列
        except RuntimeError as e:
            # 如果仍然失败，使用更强的正则化
            print(f"[WARN] 特征分解失败，使用更强正则化: {e}")
            C_reg = C_float + 1e-3 * torch.eye(C_float.shape[0], device=C_float.device, dtype=C_float.dtype)
            evals, evecs = torch.linalg.eigh(C_reg)
        
        evecs = evecs.to(orig_dtype)  # 转回原精度
        return evecs[:, -k:]  # 取最大的 k 个

    def compute_evidence_subspace(
        self, 
        vis_states: Tensor, 
        last_txt: Tensor
    ) -> Tensor:
        """
        计算证据子空间 U（基于视觉-文本相似度加权 SVD）
        
        使用SVD方法直接提取主成分，避免构造大的协方差矩阵。
        这在样本数远小于维度时（n_v << d）更稳定、更高效。
        
        Args:
            vis_states: [n_v, d] 视觉 token 隐藏状态（n_v 通常为几百，d=4096）
            last_txt: [d] 当前文本末端隐藏状态
        Returns:
            U: [d, r] 证据子空间基向量
        """
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 计算余弦相似度权重（可切换为均匀 SVD）
            if self.cfg.uniform_svd:
                w = None
            else:
                v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
                t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
                cos = (vis_states @ last_txt) / (v_norm * t_norm)
                # 使用温度平滑，温度>1 更均匀；<=0 则退化为无温度
                temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
                w = torch.softmax(cos / temp, dim=0)
            # 使用（加权/均匀）SVD直接提取主成分（避免构造4096x4096的协方差矩阵）
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U

    def compute_anti_prior_subspace(self, nonvis_txt_states: Tensor, U: Tensor) -> Tensor:
        """
        计算反先验子空间 P（在视觉补空间中的非视觉文本主成分）
        
        根据方法论：先将非视觉文本状态投影到视觉补空间 (I - UU^T)，
        再在该空间内提取主成分，确保 U^T P = 0，避免与视觉证据冲突。
        
        Args:
            nonvis_txt_states: [n_t, d] 非视觉文本 token 隐藏状态
            U: [d, r] 证据子空间基
        Returns:
            P: [d, q] 反先验子空间基向量（位于视觉补空间）
        """
        if nonvis_txt_states.numel() == 0 or self.cfg.rank_prior == 0:
            d = nonvis_txt_states.shape[-1] if nonvis_txt_states.numel() > 0 else (U.shape[0] if U.numel() > 0 else 4096)
            return nonvis_txt_states.new_zeros((d, 0))
        
        with torch.no_grad():
            # 先投影到视觉补空间：T_perp = (I - UU^T) T（可关闭）
            if (not self.cfg.no_complement) and U.numel() > 0:
                # 使用等价高效形式避免显式构造 I：T - (T U) U^T
                TU = nonvis_txt_states @ U            # [n_t, r]
                txt_in_complement = nonvis_txt_states - (TU @ U.T)  # [n_t, d]
            else:
                txt_in_complement = nonvis_txt_states
            
            # 在补空间中提取主成分
            P = self._weighted_svd(txt_in_complement, None, self.cfg.rank_prior)
        return P

    def trust_gate(self, h: Tensor, U: Tensor) -> Tensor:
        """
        计算信任域门控值（基于证据覆盖率）
        Args:
            h: [..., d] 隐藏状态
            U: [d, r] 证据子空间
        Returns:
            gate: [...] 门控值（0=高证据，1=低证据）
        """
        if U.numel() == 0:
            return torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        
        Uh = h @ U  # [..., r] 投影到证据子空间
        num = (Uh * Uh).sum(dim=-1)
        den = (h * h).sum(dim=-1) + self.cfg.eps
        evidence_score = (num / den).clamp(0.0, 1.0)  # 证据覆盖率
        gate = self.cfg.kappa * (1.0 - evidence_score)  # 反向门控
        return gate

    def edit_text_hidden(
        self, 
        h_last_txt: Tensor, 
        U: Tensor, 
        P: Tensor
    ) -> ECSEReturn:
        """
        对文本末端隐藏状态进行 ECSE 编辑
        Args:
            h_last_txt: [d] 当前生成 token 的隐藏状态
            U: [d, r] 证据子空间
            P: [d, q] 反先验子空间
        Returns:
            ECSEReturn: 编辑结果 + 证书
        """
        # 分解 h 到 U、P 与剩余子空间
        if U.numel() > 0:
            h_U = U @ (U.T @ h_last_txt)
        else:
            h_U = torch.zeros_like(h_last_txt)

        if P.numel() > 0:
            h_P = P @ (P.T @ h_last_txt)
        else:
            h_P = torch.zeros_like(h_last_txt)

        h_R = h_last_txt - h_U - h_P

        # 计算 VCR/PCR（并进行稳健化限幅）
        h_norm_sq = (h_last_txt * h_last_txt).sum() + self.cfg.eps
        vcr_raw = (h_U * h_U).sum() / h_norm_sq
        pcr_raw = (h_P * h_P).sum() / h_norm_sq
        vcr = torch.clamp(vcr_raw, min=self.cfg.vcr_floor, max=1.0)
        pcr = torch.clamp(pcr_raw, min=0.0, max=self.cfg.pcr_ceiling)

        # 计算编辑强度
        if self.cfg.use_fixed_strengths:
            # 固定强度（覆盖一切自适应机制）
            lambda_n = torch.tensor(float(max(0.0, self.cfg.fixed_lambda_n)),
                                    device=h_last_txt.device, dtype=h_last_txt.dtype)
            lambda_p = torch.tensor(float(max(0.0, self.cfg.fixed_lambda_p)),
                                    device=h_last_txt.device, dtype=h_last_txt.dtype)
        else:
            # 自适应强度（闭式最小范数编辑）
            if self.cfg.only_anti_prior:
                lambda_n = torch.zeros((), device=h_last_txt.device, dtype=h_last_txt.dtype)
            elif self.cfg.no_gating:
                # 关闭门控：使用常数型 λ_n，不依赖证据覆盖率
                lambda_n = torch.tensor(float(max(0.0, self.cfg.kappa)),
                                        device=h_last_txt.device, dtype=h_last_txt.dtype)
            else:
                lambda_n = self.cfg.kappa * (1.0 - vcr) / (vcr + self.cfg.eps)
            if self.cfg.only_residual:
                lambda_p = torch.zeros((), device=h_last_txt.device, dtype=h_last_txt.dtype)
            elif pcr < self.cfg.pcr_threshold:
                lambda_p = torch.zeros((), device=h_last_txt.device, dtype=h_last_txt.dtype)
            else:
                lambda_p = self.cfg.lambda_prior * pcr / (1.0 - pcr + self.cfg.eps)

        # 限幅，防止过度收缩
        lambda_n = torch.clamp(lambda_n, min=0.0, max=self.cfg.lambda_n_max)
        lambda_p = torch.clamp(lambda_p, min=0.0, max=self.cfg.lambda_p_max)

        # 闭式缩放
        h_edited = h_U + h_P / (1.0 + lambda_n + lambda_p) + h_R / (1.0 + lambda_n)

        # 可选：恢复范数尺度，避免 logits 过度缩小
        if self.cfg.norm_preserve:
            h_prime_norm = torch.sqrt((h_edited * h_edited).sum() + self.cfg.eps)
            h_orig_norm = torch.sqrt((h_last_txt * h_last_txt).sum() + self.cfg.eps)
            scale = (h_orig_norm / h_prime_norm).pow(self.cfg.norm_beta)
            h_edited = h_edited * scale

        # 混合原始状态以提高稳健性
        tau = float(self.cfg.blend_tau)
        tau = max(0.0, min(1.0, tau))
        if tau < 1.0:
            h_edited = (1.0 - tau) * h_last_txt + tau * h_edited

        # 记录指标；gate 用等效缩放强度表征
        gate = lambda_n / (1.0 + lambda_n)

        return ECSEReturn(
            h_edited=h_edited,
            gate=gate,
            ecr=vcr,
            epc=pcr,
            U=U,
            P=P
        )

    def clean_visual_tokens(self, vis_states: Tensor, U: Tensor) -> Tensor:
        """
        视觉 token 清洗：投影到证据子空间 v <- UU^T v
        Args:
            vis_states: [n_v, d] 视觉 token 隐藏状态
            U: [d, r] 证据子空间
        Returns:
            vis_cleaned: [n_v, d] 清洗后的视觉状态
        """
        if U.numel() == 0 or vis_states.numel() == 0:
            return vis_states
        return (vis_states @ U) @ U.T

