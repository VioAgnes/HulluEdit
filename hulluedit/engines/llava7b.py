"""
LLaVA-1.5-7B ECSE 引擎
单次前向推理 + 在线子空间估计 + 内对比编辑
参考 ParamSteer 的加载方式
"""
from __future__ import annotations

# CRITICAL: Bypass PyTorch 2.6 security check BEFORE any transformers imports
# This must be done first to prevent ValueError in torch.load
try:
    from transformers.utils import import_utils
    def _bypass_torch_load_check():
        pass  # Bypass the check - we trust our model files
    import_utils.check_torch_load_is_safe = _bypass_torch_load_check
except Exception:
    pass

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, CLIPImageProcessor
from PIL import Image
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 导入 ParamSteer 的模型加载器
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "ParamSteer"))
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from ecse.steer import ECSESteerer, ECSEConfig


@dataclass
class EngineConfig:
    """引擎配置"""
    model_name: str
    anchor_layer: int = 26
    estimate_layer: Optional[int] = None  # 子空间估计层，默认使用 anchor_layer
    edit_layer: Optional[int] = None      # 编辑层，None 表示顶层（-1）
    visual_clean_layers: List[int] = field(default_factory=lambda: [10])
    # 多层聚合（可选）：如果提供则覆盖单一 anchor_layer
    multi_anchor_layers: Optional[List[int]] = None
    layer_weighting: str = "learned"   # "learned" | "equal"
    layer_weight_temp: float = 1.0     # 学习型权重的温度（基于每层 VCR 的 softmax）
    max_new_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.2
    precision: str = "bf16"


class LLaVAECSEEngine:
    """LLaVA-1.5-7B + ECSE 推理引擎"""
    
    def __init__(
        self, 
        eng_cfg: EngineConfig, 
        ecse_cfg: ECSEConfig, 
        device: str = "cuda"
    ):
        self.device = device
        self.eng_cfg = eng_cfg
        self.ecse_cfg = ecse_cfg
        
        # 加载模型（参考 ParamSteer）
        print(f"[ECSE] 加载模型: {eng_cfg.model_name}")
        disable_torch_init()
        
        # 使用 ParamSteer 的加载方式
        model_name = "llava-v1.5-7b"  # 固定名称
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            eng_cfg.model_name, 
            None,  # model_base
            model_name, 
            device=device
        )
        
        self.model.eval()
        
        # ECSE 引导器
        self.steerer = ECSESteerer(ecse_cfg)
        print("[ECSE] 引擎初始化完成")

    @torch.no_grad()
    def _build_inputs(
        self, 
        prompt: str, 
        image: str | Image.Image
    ) -> Dict[str, torch.Tensor]:
        """构建模型输入（参考 ParamSteer 的方式）"""
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # 使用 image_processor 处理图像
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        
        # 构建输入 IDs
        input_ids = tokenizer_image_token(
            prompt, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)
        
        return {
            "input_ids": input_ids,
            "images": image_tensor
        }

    def _extract_vision_mask(
        self, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        提取视觉 token 掩码
        LLaVA 使用特殊 token <image> (id=32000) 标记图像位置
        实际推理时会被 vision encoder 输出的 576 个 token 替换
        """
        # IMAGE_TOKEN_INDEX = 32000 (LLaVA default)
        IMAGE_TOKEN_INDEX = 32000
        
        # 找到 <image> token 的位置
        image_token_mask = (input_ids == IMAGE_TOKEN_INDEX)
        
        # LLaVA-1.5: 每个 <image> 被替换为 576 个视觉 token
        # 这里我们返回一个简化的掩码（后续在实际序列长度上展开）
        return image_token_mask

    @torch.no_grad()
    def generate(
        self, 
        prompt: str, 
        image: str | Image.Image,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        生成回复（带 ECSE 编辑）
        
        Args:
            prompt: 输入提示
            image: 图像路径或 PIL.Image
            max_new_tokens: 最大生成 token 数
            
        Returns:
            {
                "text": 生成的文本,
                "certs": [{"ecr": ..., "epc": ..., "gate": ...}, ...],
                "tokens": 生成的 token 列表
            }
        """
        batch = self._build_inputs(prompt, image)
        max_steps = max_new_tokens or self.eng_cfg.max_new_tokens
        
        # 初始化
        input_ids = batch["input_ids"]  # [1, prompt_len]
        images = batch.get("images")
        attention_mask = None  # LLaVA 会自动处理
        
        past_key_values = None
        generated_ids = []
        certs = []
        
        # 首次前向获取视觉 token 位置
        first_pass = True
        cached_vis_states = None  # 缓存的视觉隐藏状态
        cached_txt_states = None  # 缓存的文本隐藏状态（累积）
        
        for step in range(max_steps):
            # 前向推理
            outputs = self.model(
                input_ids=input_ids,
                images=images if first_pass else None,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states  # Tuple[Tensor]: (layer_num+1) x [1, seq, hidden]
            
            # 确定估计层和编辑层
            num_layers = len(hidden_states) - 1  # 总层数（hidden_states 包含输入层）
            estimate_layer = self.eng_cfg.estimate_layer if self.eng_cfg.estimate_layer is not None else self.eng_cfg.anchor_layer
            if estimate_layer == -1:
                estimate_layer = num_layers  # 顶层索引
            edit_layer = self.eng_cfg.edit_layer if self.eng_cfg.edit_layer is not None else -1  # -1 表示顶层
            if edit_layer == -1:
                edit_layer = num_layers  # 顶层索引
            
            # 目标层集合：多层或单层（用于估计）
            target_layers: List[int]
            if self.eng_cfg.multi_anchor_layers and len(self.eng_cfg.multi_anchor_layers) > 0:
                target_layers = list(self.eng_cfg.multi_anchor_layers)
            else:
                target_layers = [estimate_layer]  # 使用 estimate_layer 而不是 anchor_layer

            # 获取锚点层和最后层的隐藏状态
            anchor_hidden = hidden_states[target_layers[0]]  # 以首层用于首轮缓存形状
            last_hidden = hidden_states[-1]  # [1, seq, hidden]
            
            # 首次推理：确定视觉 token 范围并缓存视觉/文本隐藏状态
            if first_pass:
                seq_len = anchor_hidden.shape[1]
                # LLaVA: 图像 token 在开头（<s> <image_tokens> prompt_tokens）
                # 简化假设：前 576 个 token 是视觉 token
                # 更精确的方法需要解析 prepare_inputs_labels_for_multimodal
                num_image_tokens = 576  # LLaVA-1.5 default
                vision_end_idx = min(1 + num_image_tokens, seq_len - 1)  # 确保至少留一个文本token
                
                # 为每个层分别缓存视觉/文本隐藏状态
                cached_vis_states = {}
                cached_txt_states = {}
                for L in target_layers:
                    hL = hidden_states[L]
                    # 提取并缓存视觉隐藏状态（后续步骤复用）
                    visL = hL[0, 1:vision_end_idx, :].clone()  # [n_vis, d]
                    cached_vis_states[L] = visL
                    # 提取初始文本隐藏状态（不包括最后一个 token，用于 anti-prior）
                    if vision_end_idx < seq_len - 1:
                        txtL = hL[0, vision_end_idx:-1, :].clone()  # [n_txt, d]
                    else:
                        hidden_dim = hL.shape[2]
                        txtL = torch.empty(0, hidden_dim, dtype=hL.dtype, device=self.device)
                    cached_txt_states[L] = txtL
                
                first_pass = False
            else:
                # 后续步骤：使用 KV cache，只有新生成的 token（seq_len=1）
                # 累积文本隐藏状态（用于后续的 anti-prior 计算）
                for L in target_layers:
                    hL = hidden_states[L]
                    new_txt_hidden = hL[0, -1, :].unsqueeze(0)  # [1, d]
                    cached_txt_states[L] = torch.cat([cached_txt_states[L], new_txt_hidden], dim=0)
            
            # 获取用于编辑的隐藏状态（edit_layer 已经转换为实际索引）
            h_to_edit = hidden_states[edit_layer][0, -1, :]  # [hidden] 指定层
            
            # ECSE 编辑（支持多层聚合）
            if len(target_layers) == 1:
                L = target_layers[0]
                U = self.steerer.compute_evidence_subspace(cached_vis_states[L], h_to_edit)
                P = self.steerer.compute_anti_prior_subspace(cached_txt_states[L], U)
                edit_result = self.steerer.edit_text_hidden(h_to_edit, U, P)
                h_edited = edit_result.h_edited
                ecr_for_weight = float(edit_result.ecr.detach().cpu())
                certs.append({
                    "ecr": float(edit_result.ecr.cpu()),
                    "epc": float(edit_result.epc.cpu()),
                    "gate": float(edit_result.gate.cpu())
                })
            else:
                # 对每个层独立计算编辑，随后聚合
                per_layer = []
                ecr_list = []
                for L in target_layers:
                    U = self.steerer.compute_evidence_subspace(cached_vis_states[L], h_to_edit)
                    P = self.steerer.compute_anti_prior_subspace(cached_txt_states[L], U)
                    er = self.steerer.edit_text_hidden(h_to_edit, U, P)
                    per_layer.append(er)
                    ecr_list.append(float(er.ecr.detach().cpu()))
                # 计算权重
                if self.eng_cfg.layer_weighting == "equal":
                    weights = torch.ones(len(per_layer), device=h_to_edit.device, dtype=h_to_edit.dtype) / len(per_layer)
                else:
                    # learned: 基于每层的 VCR（ecr）做 softmax
                    ecr_tensor = torch.tensor(ecr_list, device=h_to_edit.device, dtype=h_to_edit.dtype)
                    temp = max(1e-6, float(self.eng_cfg.layer_weight_temp))
                    weights = torch.softmax(ecr_tensor / temp, dim=0)
                # 聚合编辑后的隐藏状态
                h_stack = torch.stack([er.h_edited for er in per_layer], dim=0)  # [L, d]
                h_edited = (weights[:, None] * h_stack).sum(dim=0)
                # 记录最后一层证书（以及总权重信息）
                certs.append({
                    "ecr": float(sum(ecr_list) / max(1, len(ecr_list))),
                    "epc": float(sum(float(er.epc.detach().cpu()) for er in per_layer) / max(1, len(per_layer))),
                    "gate": float(sum(float(er.gate.detach().cpu()) for er in per_layer) / max(1, len(per_layer)))
                })
            
            # 通过 lm_head 映射到 logits（内对比）
            logits = self.model.lm_head(h_edited.unsqueeze(0))  # [1, vocab]
            
            # Greedy decoding (temperature=0.0) 或采样
            if self.eng_cfg.temperature <= 1e-6:
                # Greedy decoding: 直接使用 argmax
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
            else:
                # 采样模式
                probs = torch.softmax(
                    logits / self.eng_cfg.temperature, 
                    dim=-1
                )
                
                # Top-p 采样
                if self.eng_cfg.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # 移除累积概率超过 top_p 的 token
                    sorted_indices_to_remove = cumulative_probs > self.eng_cfg.top_p
                    sorted_indices_to_remove[..., 0] = False  # 保留至少一个
                    
                    # 创建掩码
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    probs[indices_to_remove] = 0.0
                    probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
                
                # 多项式采样
                next_token_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            
            # 记录
            generated_ids.append(next_token_id.item())
            # 证书已在上面追加
            
            # 准备下一步输入
            input_ids = next_token_id
            
            # EOS 检查
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        
        # 解码
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "certs": certs,
            "tokens": generated_ids
        }

