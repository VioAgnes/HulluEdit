"""
MME (Multi-Modal Evaluation) 评测
基于 ECSE 方法，评估 LLaVA 和 MiniGPT-4 在 MME 数据集上的表现

参考 DeCo/mme_llava.py 的实现
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml
import torch
from PIL import Image
from tqdm import tqdm

from ecse.steer import ECSEConfig


def _load_mme_questions(question_file: str, image_folder: str) -> List[Dict[str, Any]]:
    """
    读取 MME 测试文件，返回问题列表
    
    Args:
        question_file: MME 测试文件路径（JSONL 格式）
        image_folder: MME 图像根目录
    
    Returns:
        问题列表，每个元素包含：question_id, image, prompt, text
    """
    image_folder_path = Path(image_folder).expanduser()
    if not image_folder_path.exists():
        raise FileNotFoundError(f"MME 图像目录不存在: {image_folder_path}")
    
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 构建完整图像路径
                image_path = item.get("image", "")
                if image_path.startswith("/"):
                    # 绝对路径
                    full_image_path = image_path
                else:
                    # 相对路径
                    full_image_path = str(image_folder_path / image_path)
                
                questions.append({
                    "question_id": item.get("question_id"),
                    "image": image_path,
                    "image_path": full_image_path,
                    "prompt": item.get("prompt", ""),
                    "text": item.get("text", ""),  # 可能包含参考答案
                })
            except Exception as e:
                print(f"解析行失败: {line}, 错误: {e}")
                continue
    
    if not questions:
        raise RuntimeError("MME 测试集解析为空，请检查 question_file 内容")
    
    return questions


def recorder(output: str) -> str:
    """
    将模型输出转换为 Yes/No 答案
    参考 DeCo/mme_llava.py 的实现
    """
    NEG_WORDS = ["No", "not", "no", "NO"]
    
    output = output.replace('.', '')
    output = output.replace(',', '')
    words = output.split(' ')
    
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def main():
    parser = argparse.ArgumentParser(description="MME 评测（基于 ECSE）")
    parser.add_argument("--model_name", type=str, default="LLaVA-7B",
                        help="模型名称，用于输出文件命名")
    parser.add_argument("--question_file", type=str, required=True,
                        help="MME 测试文件路径（JSONL 格式）")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="MME 图像根目录")
    parser.add_argument("--config", type=str, required=True,
                        help="ECSE 配置文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="最大样本数（None 表示全部）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    print(f"[INFO] 加载配置文件: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    # 加载问题
    print(f"[INFO] 加载 MME 测试集: {args.question_file}")
    questions = _load_mme_questions(args.question_file, args.image_folder)
    
    if args.num_samples and args.num_samples < len(questions):
        print(f"[INFO] 采样 {args.num_samples} 个问题（总共 {len(questions)} 个）")
        questions = random.sample(questions, args.num_samples)
    
    print(f"[INFO] 共 {len(questions)} 个问题待评测")
    
    # 初始化 ECSE 配置
    print(f"[INFO] 初始化 ECSE 配置")
    ecse_cfg = ECSEConfig(
        rank_evidence=config_dict.get("rank_evidence", 8),
        rank_prior=config_dict.get("rank_prior", 5),
        kappa=config_dict.get("kappa", 0.50),
        lambda_prior=config_dict.get("lambda_prior", 0.22),
        eps=config_dict.get("eps", 1e-6),
        lambda_n_max=config_dict.get("lambda_n_max", 3.2),
        lambda_p_max=config_dict.get("lambda_p_max", 3.8),
        vcr_floor=config_dict.get("vcr_floor", 0.045),
        pcr_ceiling=config_dict.get("pcr_ceiling", 0.92),
        pcr_threshold=config_dict.get("pcr_threshold", 0.015),
        blend_tau=config_dict.get("blend_tau", 0.76),
        norm_preserve=config_dict.get("norm_preserve", True),
        norm_beta=config_dict.get("norm_beta", 0.74),
        weight_temp=config_dict.get("weight_temp", 1.15),
    )
    
    # 加载模型
    model_name = config_dict.get("model_name", args.model_name)
    print(f"[INFO] 加载模型: {model_name}")
    from ecse.engines.llava7b import LLaVAECSEEngine, EngineConfig
    from ecse.engines.minigpt4 import MiniGPT4ECSEEngine, MiniGPT4EngineConfig
    
    model_name_lower = args.model_name.lower()
    if "llava" in model_name_lower:
        eng_cfg = EngineConfig(
            model_name=model_name,
            anchor_layer=config_dict.get("anchor_layer", 28),
            visual_clean_layers=config_dict.get("visual_clean_layers", [10]),
            max_new_tokens=config_dict.get("max_new_tokens", 128),
            top_p=config_dict.get("top_p", 0.90),
            temperature=config_dict.get("temperature", 0.12),
            precision=config_dict.get("precision", "bf16"),
        )
        engine = LLaVAECSEEngine(
            eng_cfg=eng_cfg,
            ecse_cfg=ecse_cfg,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    elif "minigpt" in model_name_lower:
        eng_cfg = MiniGPT4EngineConfig(
            cfg_path=config_dict.get("minigpt4_cfg_path"),
            anchor_layer=config_dict.get("anchor_layer", 28),
            max_new_tokens=config_dict.get("max_new_tokens", 128),
            top_p=config_dict.get("top_p", 0.90),
            temperature=config_dict.get("temperature", 0.12),
        )
        engine = MiniGPT4ECSEEngine(
            eng_cfg=eng_cfg,
            ecse_cfg=ecse_cfg,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"不支持的模型: {args.model_name}")
    
    # 生成答案
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.model_name.lower()}_mme_answers_{timestamp}.jsonl"
    
    print(f"[INFO] 开始生成答案，输出到: {output_file}")
    
    results = []
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(questions, desc="MME 评测"):
            question_id = item["question_id"]
            image_path = item["image_path"]
            prompt = item["prompt"]
            
            # 检查图像是否存在
            if not Path(image_path).exists():
                print(f"[WARNING] 图像不存在: {image_path}")
                continue
            
            try:
                # 生成答案（根据引擎类型使用不同的接口）
                if "llava" in model_name_lower:
                    # LLaVA 接受 PIL Image 或路径
                    image = Image.open(image_path).convert("RGB")
                    result_dict = engine.generate(prompt, image)
                    output = result_dict.get("output", "")
                elif "minigpt" in model_name_lower:
                    # MiniGPT-4 接受路径
                    result_dict = engine.generate(prompt, image_path)
                    output = result_dict.get("output", "")
                else:
                    raise ValueError(f"不支持的模型: {args.model_name}")
                
                # 转换为 Yes/No
                answer = recorder(output)
                
                # 保存结果
                result = {
                    "question_id": question_id,
                    "prompt": prompt,
                    "text": answer,
                    "model_id": args.model_name,
                    "image": item["image"],
                    "metadata": {
                        "raw_output": output
                    }
                }
                
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                
            except Exception as e:
                print(f"[ERROR] 处理问题失败 {question_id}: {e}")
                continue
    
    print(f"[INFO] 评测完成，共生成 {len(results)} 个答案")
    print(f"[INFO] 结果保存在: {output_file}")
    
    # 输出统计信息
    yes_count = sum(1 for r in results if r["text"] == "Yes")
    no_count = sum(1 for r in results if r["text"] == "No")
    print(f"[INFO] 答案统计: Yes={yes_count}, No={no_count}")
    
    return output_file


if __name__ == "__main__":
    main()

