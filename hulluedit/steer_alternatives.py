"""
视觉子空间构建的替代方案实现
用于快速测试和对比不同方案的效果
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional
from ecse.steer import ECSESteerer, ECSEConfig


class AlternativeSteerer(ECSESteerer):
    """扩展的ECSE引导器，支持多种视觉子空间构建方案"""
    
    def __init__(self, config: ECSEConfig, method: str = "cosine"):
        """
        Args:
            config: ECSE配置
            method: 视觉子空间构建方法
                - "cosine": 当前方法（余弦相似度加权）
                - "standard": 方案1 - 标准PCA
                - "variance": 方案2 - 方差加权
                - "multi_text": 方案3 - 多文本token
                - "hybrid": 方案4 - 混合权重
                - "adaptive": 方案6 - 自适应rank
        """
        super().__init__(config)
        self.method = method
    
    def compute_evidence_subspace(
        self, 
        vis_states: Tensor, 
        last_txt: Tensor,
        txt_states: Optional[Tensor] = None
    ) -> Tensor:
        """根据method选择不同的构建方案"""
        if self.method == "cosine":
            return self._compute_cosine_weighted(vis_states, last_txt)
        elif self.method == "standard":
            return self._compute_standard_pca(vis_states, last_txt)
        elif self.method == "variance":
            return self._compute_variance_weighted(vis_states, last_txt)
        elif self.method == "multi_text":
            if txt_states is None:
                # 回退到单文本
                return self._compute_cosine_weighted(vis_states, last_txt)
            return self._compute_multi_text(vis_states, txt_states)
        elif self.method == "hybrid":
            return self._compute_hybrid(vis_states, last_txt)
        elif self.method == "adaptive":
            return self._compute_adaptive_rank(vis_states, last_txt)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_cosine_weighted(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案0：当前方法 - 余弦相似度加权"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_standard_pca(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案1：标准PCA，无权重"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        # 直接SVD，无权重
        U = self._weighted_svd(vis_states, w=None, k=self.cfg.rank_evidence)
        return U
    
    def _compute_variance_weighted(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案2：基于方差的加权PCA"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 计算每个视觉token的方差
            vis_mean = vis_states.mean(dim=0, keepdim=True)
            vis_var = ((vis_states - vis_mean) ** 2).sum(dim=1)  # [n_v]
            
            # 归一化为权重
            w = vis_var / (vis_var.sum() + self.cfg.eps)
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_multi_text(
        self, vis_states: Tensor, txt_states: Tensor
    ) -> Tensor:
        """方案3：使用多个文本token的相似度"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = txt_states.shape[-1] if txt_states.numel() > 0 else 4096
            return txt_states.new_zeros((d, 0))
        
        with torch.no_grad():
            # 使用最后N个文本token
            n_text = min(5, txt_states.shape[0])
            txt_avg = txt_states[-n_text:].mean(dim=0)  # [d]
            
            # 计算平均相似度
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(txt_avg) + self.cfg.eps
            cos = (vis_states @ txt_avg) / (v_norm * t_norm)
            
            # 或者：计算与每个文本token的相似度，然后平均
            # t_norms = torch.linalg.norm(txt_states[-n_text:], dim=1) + self.cfg.eps
            # cos_all = (vis_states @ txt_states[-n_text:].T) / (v_norm[:, None] * t_norms[None, :])
            # cos = cos_all.mean(dim=1)
            
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_hybrid(
        self, vis_states: Tensor, last_txt: Tensor, alpha: float = 0.7
    ) -> Tensor:
        """方案4：混合权重（相似度 + 方差）"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 1. 余弦相似度权重
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w_sim = torch.softmax(cos / temp, dim=0)
            
            # 2. 方差权重
            vis_mean = vis_states.mean(dim=0, keepdim=True)
            vis_var = ((vis_states - vis_mean) ** 2).sum(dim=1)
            w_var = vis_var / (vis_var.sum() + self.cfg.eps)
            
            # 3. 混合
            w = alpha * w_sim + (1 - alpha) * w_var
            w = w / (w.sum() + self.cfg.eps)  # 重新归一化
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_adaptive_rank(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案6：自适应rank选择"""
        if vis_states.numel() == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 先做SVD，查看奇异值
            vis_centered = vis_states - vis_states.mean(dim=0, keepdim=True)
            
            if vis_centered.abs().max() < 1e-10:
                # 如果中心化后接近零，使用固定rank
                rank_adaptive = self.cfg.rank_evidence
            else:
                try:
                    _, s, _ = torch.linalg.svd(vis_centered.float(), full_matrices=False)
                    
                    # 方法1：保留累积能量 > 0.95的主成分
                    cumsum = torch.cumsum(s, dim=0)
                    cumsum_norm = cumsum / (cumsum[-1] + 1e-12)
                    rank_adaptive = (cumsum_norm < 0.95).sum().item() + 1
                    rank_adaptive = min(rank_adaptive, len(s), self.cfg.rank_evidence * 2)
                    
                    # 限制在合理范围
                    rank_adaptive = max(1, min(rank_adaptive, len(s), 32))
                except Exception:
                    # 如果SVD失败，使用固定rank
                    rank_adaptive = self.cfg.rank_evidence
            
            if rank_adaptive == 0:
                d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
                return last_txt.new_zeros((d, 0))
            
            # 使用选择的rank重新计算（可以加权重）
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            
            U = self._weighted_svd(vis_states, w, rank_adaptive)
        return U


# 为了兼容性，重写compute_evidence_subspace方法
def patch_steerer_method(steerer: ECSESteerer, method: str, alpha: float = 0.7):
    """
    动态替换steerer的compute_evidence_subspace方法
    
    Args:
        steerer: ECSESteerer实例
        method: 方法名称
        alpha: 混合权重参数（仅用于hybrid方法）
    """
    if method == "cosine":
        # 使用原始方法
        return
    
    alt_steerer = AlternativeSteerer(steerer.cfg, method)
    
    if method == "standard":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_standard_pca(vis_states, last_txt)
    elif method == "variance":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_variance_weighted(vis_states, last_txt)
    elif method == "multi_text":
        # 需要修改调用方式，传入txt_states
        # 这里先不实现，需要修改引擎代码
        raise NotImplementedError("multi_text方法需要修改引擎代码以传入txt_states")
    elif method == "hybrid":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_hybrid(vis_states, last_txt, alpha)
    elif method == "adaptive":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_adaptive_rank(vis_states, last_txt)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 替换方法
    steerer.compute_evidence_subspace = new_method.__get__(steerer, type(steerer))

视觉子空间构建的替代方案实现
用于快速测试和对比不同方案的效果
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional
from ecse.steer import ECSESteerer, ECSEConfig


class AlternativeSteerer(ECSESteerer):
    """扩展的ECSE引导器，支持多种视觉子空间构建方案"""
    
    def __init__(self, config: ECSEConfig, method: str = "cosine"):
        """
        Args:
            config: ECSE配置
            method: 视觉子空间构建方法
                - "cosine": 当前方法（余弦相似度加权）
                - "standard": 方案1 - 标准PCA
                - "variance": 方案2 - 方差加权
                - "multi_text": 方案3 - 多文本token
                - "hybrid": 方案4 - 混合权重
                - "adaptive": 方案6 - 自适应rank
        """
        super().__init__(config)
        self.method = method
    
    def compute_evidence_subspace(
        self, 
        vis_states: Tensor, 
        last_txt: Tensor,
        txt_states: Optional[Tensor] = None
    ) -> Tensor:
        """根据method选择不同的构建方案"""
        if self.method == "cosine":
            return self._compute_cosine_weighted(vis_states, last_txt)
        elif self.method == "standard":
            return self._compute_standard_pca(vis_states, last_txt)
        elif self.method == "variance":
            return self._compute_variance_weighted(vis_states, last_txt)
        elif self.method == "multi_text":
            if txt_states is None:
                # 回退到单文本
                return self._compute_cosine_weighted(vis_states, last_txt)
            return self._compute_multi_text(vis_states, txt_states)
        elif self.method == "hybrid":
            return self._compute_hybrid(vis_states, last_txt)
        elif self.method == "adaptive":
            return self._compute_adaptive_rank(vis_states, last_txt)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_cosine_weighted(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案0：当前方法 - 余弦相似度加权"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_standard_pca(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案1：标准PCA，无权重"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        # 直接SVD，无权重
        U = self._weighted_svd(vis_states, w=None, k=self.cfg.rank_evidence)
        return U
    
    def _compute_variance_weighted(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案2：基于方差的加权PCA"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 计算每个视觉token的方差
            vis_mean = vis_states.mean(dim=0, keepdim=True)
            vis_var = ((vis_states - vis_mean) ** 2).sum(dim=1)  # [n_v]
            
            # 归一化为权重
            w = vis_var / (vis_var.sum() + self.cfg.eps)
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_multi_text(
        self, vis_states: Tensor, txt_states: Tensor
    ) -> Tensor:
        """方案3：使用多个文本token的相似度"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = txt_states.shape[-1] if txt_states.numel() > 0 else 4096
            return txt_states.new_zeros((d, 0))
        
        with torch.no_grad():
            # 使用最后N个文本token
            n_text = min(5, txt_states.shape[0])
            txt_avg = txt_states[-n_text:].mean(dim=0)  # [d]
            
            # 计算平均相似度
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(txt_avg) + self.cfg.eps
            cos = (vis_states @ txt_avg) / (v_norm * t_norm)
            
            # 或者：计算与每个文本token的相似度，然后平均
            # t_norms = torch.linalg.norm(txt_states[-n_text:], dim=1) + self.cfg.eps
            # cos_all = (vis_states @ txt_states[-n_text:].T) / (v_norm[:, None] * t_norms[None, :])
            # cos = cos_all.mean(dim=1)
            
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_hybrid(
        self, vis_states: Tensor, last_txt: Tensor, alpha: float = 0.7
    ) -> Tensor:
        """方案4：混合权重（相似度 + 方差）"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 1. 余弦相似度权重
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w_sim = torch.softmax(cos / temp, dim=0)
            
            # 2. 方差权重
            vis_mean = vis_states.mean(dim=0, keepdim=True)
            vis_var = ((vis_states - vis_mean) ** 2).sum(dim=1)
            w_var = vis_var / (vis_var.sum() + self.cfg.eps)
            
            # 3. 混合
            w = alpha * w_sim + (1 - alpha) * w_var
            w = w / (w.sum() + self.cfg.eps)  # 重新归一化
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_adaptive_rank(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """方案6：自适应rank选择"""
        if vis_states.numel() == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 先做SVD，查看奇异值
            vis_centered = vis_states - vis_states.mean(dim=0, keepdim=True)
            
            if vis_centered.abs().max() < 1e-10:
                # 如果中心化后接近零，使用固定rank
                rank_adaptive = self.cfg.rank_evidence
            else:
                try:
                    _, s, _ = torch.linalg.svd(vis_centered.float(), full_matrices=False)
                    
                    # 方法1：保留累积能量 > 0.95的主成分
                    cumsum = torch.cumsum(s, dim=0)
                    cumsum_norm = cumsum / (cumsum[-1] + 1e-12)
                    rank_adaptive = (cumsum_norm < 0.95).sum().item() + 1
                    rank_adaptive = min(rank_adaptive, len(s), self.cfg.rank_evidence * 2)
                    
                    # 限制在合理范围
                    rank_adaptive = max(1, min(rank_adaptive, len(s), 32))
                except Exception:
                    # 如果SVD失败，使用固定rank
                    rank_adaptive = self.cfg.rank_evidence
            
            if rank_adaptive == 0:
                d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
                return last_txt.new_zeros((d, 0))
            
            # 使用选择的rank重新计算（可以加权重）
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            
            U = self._weighted_svd(vis_states, w, rank_adaptive)
        return U


# 为了兼容性，重写compute_evidence_subspace方法
def patch_steerer_method(steerer: ECSESteerer, method: str, alpha: float = 0.7):
    """
    动态替换steerer的compute_evidence_subspace方法
    
    Args:
        steerer: ECSESteerer实例
        method: 方法名称
        alpha: 混合权重参数（仅用于hybrid方法）
    """
    if method == "cosine":
        # 使用原始方法
        return
    
    alt_steerer = AlternativeSteerer(steerer.cfg, method)
    
    if method == "standard":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_standard_pca(vis_states, last_txt)
    elif method == "variance":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_variance_weighted(vis_states, last_txt)
    elif method == "multi_text":
        # 需要修改调用方式，传入txt_states
        # 这里先不实现，需要修改引擎代码
        raise NotImplementedError("multi_text方法需要修改引擎代码以传入txt_states")
    elif method == "hybrid":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_hybrid(vis_states, last_txt, alpha)
    elif method == "adaptive":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_adaptive_rank(vis_states, last_txt)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 替换方法
    steerer.compute_evidence_subspace = new_method.__get__(steerer, type(steerer))

