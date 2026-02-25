"""
MiniGPT-4 推理引擎（用于 POPE/CHAIR 评测，集成 ECSE 编辑）

使用 MODELS/MiniGPT-4 下的官方实现，参考 DeCo 项目的实现方式。
不依赖 sjc/Nullu 文件夹。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import sys
import torch
from PIL import Image

# 添加 MODELS/MiniGPT-4 到路径（必须在最前面，避免与其他版本冲突）
_MINIGPT4_ROOT = os.environ.get("MINIGPT4_ROOT", "/data/home/scyb531/MODELS/MiniGPT-4")
_MINIGPT4_ROOT = os.path.abspath(_MINIGPT4_ROOT)
if _MINIGPT4_ROOT not in sys.path:
    sys.path.insert(0, _MINIGPT4_ROOT)

try:
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import (
        Chat as MiniChat,
        CONV_VISION_LLama2,
        CONV_VISION_Vicuna0,
    )
    # 导入模块以注册模型、处理器等
    from minigpt4.datasets.builders import *
    from minigpt4.models import *
    from minigpt4.processors import *
except Exception as e:
    raise ImportError(
        f"未找到 MiniGPT-4 依赖，请确保 MINIGPT4_ROOT 指向包含 minigpt4 的仓库。"
        f"当前: {_MINIGPT4_ROOT}. 错误: {e}"
    )

from ecse.steer import ECSESteerer, ECSEConfig


@dataclass
class MiniGPT4EngineConfig:
    """MiniGPT-4 引擎配置"""
    cfg_path: str  # MiniGPT-4 评估 YAML 配置文件路径
    anchor_layer: int = 26
    max_new_tokens: int = 128
    top_p: float = 0.9
    temperature: float = 0.2
    repetition_penalty: float = 1.2  # 重复惩罚系数，用于减少重复生成
    gpu_id: int = 0


class MiniGPT4ECSEEngine:
    """MiniGPT-4 + ECSE 推理引擎（显式逐步生成并进行隐表示编辑）"""

    def __init__(self, eng_cfg: MiniGPT4EngineConfig, ecse_cfg: ECSEConfig, device: str = "cuda:0") -> None:
        self.device = device
        self.eng_cfg = eng_cfg
        self.ecse_cfg = ecse_cfg
        self.steerer = ECSESteerer(ecse_cfg)

        # 使用 Config 类加载配置（参考 DeCo 实现）
        class Args:
            def __init__(self, cfg_path, options=None):
                self.cfg_path = cfg_path
                self.options = options or []
        
        args = Args(eng_cfg.cfg_path, [])
        cfg = Config(args)
        
        # 初始化模型
        model_config = cfg.model_cfg
        model_config.device_8bit = eng_cfg.gpu_id
        
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        self.model.eval()
        
        # 获取对话模板（默认使用 CONV_VISION_Vicuna0 以匹配 DeCo）
        conv_dict = {
            'pretrain_vicuna0': CONV_VISION_Vicuna0,
            'pretrain_llama2': CONV_VISION_LLama2
        }
        self.model_type = model_config.model_type  # 保存模型类型，用于后处理
        self.conv_template = conv_dict.get(self.model_type, CONV_VISION_Vicuna0).copy()
        
        # 获取停止符号（end_sym），用于停止条件检查
        self.end_sym = model_config.get("end_sym", "###")
        # 将停止符号编码为 token IDs
        if self.end_sym:
            end_sym_ids = self.model.llama_tokenizer.encode(self.end_sym, add_special_tokens=False)
            # 如果停止符号是多个 token，只检查最后一个（通常 "###" 是一个 token，但 "</s>" 可能是多个）
            self.end_sym_token_id = end_sym_ids[-1] if end_sym_ids else None
            # 保存完整的停止符号 token 序列（用于更精确的检测）
            self.end_sym_token_ids = end_sym_ids if end_sym_ids else []
        else:
            self.end_sym_token_id = None
            self.end_sym_token_ids = []
        
        # 初始化视觉处理器
        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        
        # 初始化 Chat 对象（用于处理对话）
        self.chat = MiniChat(self.model, self.vis_processor, device=self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image_path: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        生成回答，集成 ECSE 编辑
        
        Args:
            prompt: 文本提示
            image_path: 图像路径
            max_new_tokens: 最大生成 token 数
            
        Returns:
            包含 text, certs, tokens 的字典
        """
        # 构建对话与图片上下文
        conv = self.conv_template.copy()
        img_list = []
        _ = self.chat.upload_img(image_path, conv, img_list)
        self.chat.encode_img(img_list)  # 这会将图像编码并放入 img_list[0]
        self.chat.ask(prompt, conv)

        # 准备输入嵌入与采样参数
        gen_kwargs = self.chat.answer_prepare(
            conv,
            img_list,
            max_new_tokens=max_new_tokens or self.eng_cfg.max_new_tokens,
            top_p=self.eng_cfg.top_p,
            temperature=self.eng_cfg.temperature,
        )
        inputs_embeds: torch.Tensor = gen_kwargs.pop("inputs_embeds")  # [1, seq, hidden]

        # 从 img_list 中获取视觉嵌入长度（避免重复计算）
        # img_list[0] 是 encode_img 的结果，形状为 [1, n_vis, hidden_dim]
        if len(img_list) > 0 and isinstance(img_list[0], torch.Tensor):
            n_vis = int(img_list[0].shape[1])
        else:
            # 如果 img_list 为空或格式不对，回退到重新计算（不应该发生）
            raw_image = Image.open(image_path).convert("RGB")
            image_tensor = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            image_emb, _ = self.model.encode_img(image_tensor)
            n_vis = int(image_emb.shape[1])

        past_key_values = None
        generated_ids = []
        certs = []
        first_pass = True
        cached_vis_states = None
        cached_txt_states = None
        max_steps = max_new_tokens or self.eng_cfg.max_new_tokens

        for _ in range(max_steps):
            if first_pass:
                outputs = self.model.llama_model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                # 后续步骤：使用 input_ids 和 past_key_values
                # 注意：next_token_id 的形状应该是 [1, 1]
                if next_token_id.dim() == 1:
                    next_token_id = next_token_id.unsqueeze(0)
                outputs = self.model.llama_model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states  # Tuple[Tensors]

            # 检查 anchor_layer 是否在有效范围内
            num_layers = len(hidden_states)
            if self.eng_cfg.anchor_layer >= num_layers:
                raise ValueError(
                    f"anchor_layer {self.eng_cfg.anchor_layer} 超出范围 "
                    f"(模型共有 {num_layers} 层，索引范围 0-{num_layers-1})"
                )
            
            anchor_hidden = hidden_states[self.eng_cfg.anchor_layer]  # [1, seq, d]
            last_hidden = hidden_states[-1]

            if first_pass:
                seq_len = anchor_hidden.shape[1]
                # 计算视觉嵌入的结束位置
                # inputs_embeds 的结构: [text_before_image, image_emb, text_after_image]
                # 需要找到 <ImageHere> 在 prompt 中的位置来确定视觉嵌入的边界
                # 由于 answer_prepare 使用 get_context_emb，视觉嵌入已经嵌入到 inputs_embeds 中
                # 我们使用 n_vis 作为视觉 token 的数量
                vision_end_idx = min(n_vis, seq_len)
                
                # 确保索引有效
                if vision_end_idx > seq_len:
                    vision_end_idx = seq_len
                
                # 缓存视觉隐藏状态（从 anchor_hidden 中提取视觉部分）
                if vision_end_idx > 0:
                    cached_vis_states = anchor_hidden[0, :vision_end_idx, :].clone()
                else:
                    hidden_dim = anchor_hidden.shape[2]
                    cached_vis_states = torch.empty(0, hidden_dim, dtype=anchor_hidden.dtype, device=anchor_hidden.device)
                
                # 缓存文本隐藏状态（视觉之后到最后一个位置之前的部分）
                # 最后一个位置是当前要生成的位置，不包含在文本状态中
                if vision_end_idx < seq_len - 1:
                    cached_txt_states = anchor_hidden[0, vision_end_idx:-1, :].clone()
                else:
                    hidden_dim = anchor_hidden.shape[2]
                    cached_txt_states = torch.empty(0, hidden_dim, dtype=anchor_hidden.dtype, device=anchor_hidden.device)
                
                first_pass = False
            else:
                new_txt_hidden = anchor_hidden[0, -1, :].unsqueeze(0)
                cached_txt_states = torch.cat([cached_txt_states, new_txt_hidden], dim=0)

            h_last = last_hidden[0, -1, :]

            # ECSE 编辑
            U = self.steerer.compute_evidence_subspace(cached_vis_states, h_last)
            P = self.steerer.compute_anti_prior_subspace(cached_txt_states, U)
            edit_result = self.steerer.edit_text_hidden(h_last, U, P)

            logits = self.model.llama_model.lm_head(edit_result.h_edited.unsqueeze(0))

            # 应用 repetition_penalty：惩罚最近生成的 token，减少重复
            if self.eng_cfg.repetition_penalty > 1.0 and len(generated_ids) > 0:
                # 只惩罚最近 50 个 token，避免过度惩罚
                recent_tokens = set(generated_ids[-50:])
                for token_id in recent_tokens:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= self.eng_cfg.repetition_penalty
                    else:
                        logits[0, token_id] *= self.eng_cfg.repetition_penalty

            # 采样/贪心
            if self.eng_cfg.temperature < 0.01:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                safe_temp = max(self.eng_cfg.temperature, 0.01)
                probs = torch.softmax(logits / safe_temp, dim=-1)
                if self.eng_cfg.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.eng_cfg.top_p
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0.0
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids.append(next_token_id.item())
            certs.append({
                "ecr": float(edit_result.ecr.detach().cpu()),
                "epc": float(edit_result.epc.detach().cpu()),
                "gate": float(edit_result.gate.detach().cpu()),
            })

            # 检查停止条件：EOS token 或 end_sym（如 "###"）
            if next_token_id.item() == self.model.llama_tokenizer.eos_token_id:
                break
            if self.end_sym_token_id is not None and next_token_id.item() == self.end_sym_token_id:
                break

            # 检测重复生成 "Image Content" 模式（防止模型陷入重复循环）
            if len(generated_ids) >= 20:  # 至少生成 20 个 token 后才检测
                recent_text = self.model.llama_tokenizer.decode(generated_ids[-30:], skip_special_tokens=True)
                # 检查是否重复生成 "Image Content" 或 "</Img>" 后跟 "Image Content"
                if "</Img>" in recent_text:
                    # 如果生成了 "</Img>"，检查后面是否重复出现 "Image Content"
                    parts = recent_text.split("</Img>")
                    if len(parts) > 1:
                        after_img_tag = parts[-1].strip()
                        # 如果 "</Img>" 后出现多次 "Image Content"，停止生成
                        if after_img_tag.count("Image Content") >= 2:
                            break
                # 检查是否连续多次出现 "Image Content"（即使没有 "</Img>"）
                if recent_text.count("Image Content") >= 3:
                    break

        generated_text = self.model.llama_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 后处理：根据模型类型使用不同的清理方式
        if self.end_sym:
            generated_text = generated_text.split(self.end_sym)[0]
        
        # 根据模型类型移除不同的前缀
        if self.model_type == 'pretrain_llama2':
            # Llama2 格式：移除 [/INST] 之前的内容
            if "[/INST]" in generated_text:
                generated_text = generated_text.split("[/INST]")[-1].strip()
            # 移除可能的 <s> 前缀
            if generated_text.startswith("<s>"):
                generated_text = generated_text[3:].strip()
            # 移除可能的 [INST] 标签（如果模型生成了）
            if generated_text.startswith("[INST]"):
                generated_text = generated_text[6:].strip()
        else:
            # Vicuna0 格式：移除 "Assistant:" 前缀
            if "Assistant:" in generated_text:
                generated_text = generated_text.split("Assistant:")[-1].strip()
        
        # 额外后处理：移除 "</Img>" 标签及其后的重复 "Image Content"
        if "</Img>" in generated_text:
            # 找到 "</Img>" 的位置
            img_tag_idx = generated_text.find("</Img>")
            if img_tag_idx >= 0:
                # 保留 "</Img>" 之前的内容，移除 "</Img>" 及其后的内容
                # 但如果 "</Img>" 后是正常的回答，则保留
                after_img = generated_text[img_tag_idx + 6:].strip()
                # 如果 "</Img>" 后只有重复的 "Image Content"，则移除
                if after_img and "Image Content" in after_img:
                    # 检查是否主要是重复的 "Image Content"
                    cleaned_after = after_img.replace("Image Content", "").strip()
                    if len(cleaned_after) < len(after_img) * 0.3:  # 如果 70% 以上是 "Image Content"
                        generated_text = generated_text[:img_tag_idx].strip()
                    else:
                        # 保留 "</Img>" 后的正常内容，但移除重复的 "Image Content"
                        # 只保留第一个 "Image Content" 之后的内容
                        parts = after_img.split("Image Content")
                        if len(parts) > 2:  # 如果有多个 "Image Content"
                            # 保留第一个 "Image Content" 之前和之后的内容
                            generated_text = generated_text[:img_tag_idx] + "</Img>" + parts[0] + "Image Content" + parts[-1]
        
        return {"text": generated_text, "certs": certs, "tokens": generated_ids}
