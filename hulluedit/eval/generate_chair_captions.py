#!/usr/bin/env python3
"""
生成 CHAIR 评测所需的 caption（使用 ECSE 引擎）
参考 Nullu 的 CHAIR 评测流程，确保评估一致性

输出格式：JSONL 文件，每行一个 JSON
{"image_id": 123456, "caption": "A person riding a bike..."}
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import yaml
import random
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ecse.engines.llava7b import LLaVAECSEEngine, EngineConfig
from ecse.steer import ECSEConfig
from ecse.datasets.chair_dataset import build_chair_dataset


def setup_seeds(seed: int):
    """设置随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="生成 CHAIR Caption（ECSE）")
    
    # 配置文件
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    
    # 数据集参数（可覆盖配置文件）
    parser.add_argument("--split", type=str, default=None, choices=["val", "train"])
    parser.add_argument("--sampling", type=str, default=None, choices=["first", "random"])
    parser.add_argument("--num-samples", type=int, default=None)
    
    # 输出
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    
    # 其他
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="打印每张图像的生成结果")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置文件
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置随机种子
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    setup_seeds(seed)
    
    # 数据集参数（命令行优先，否则用配置文件）
    split = args.split if args.split else cfg.get("split", "val")
    sampling = args.sampling if args.sampling else cfg.get("sampling", "random")
    num_samples = args.num_samples if args.num_samples else cfg.get("num_samples", 500)
    data_root = cfg.get("coco_root", "/data/home/scyb531/DATA/")
    
    # 输出目录
    output_dir = args.output_dir if args.output_dir else cfg.get("output_dir", "/data/home/scyb531/lyg/HulluEdit/outputs/chair")
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印配置信息
    print("=" * 80)
    print("ECSE CHAIR Caption 生成")
    print("=" * 80)
    print(f"数据集设置:")
    print(f"  Split:       {split}")
    print(f"  Sampling:    {sampling}")
    print(f"  Num Samples: {num_samples}")
    print(f"  Seed:        {seed}")
    print("=" * 80)
    print(f"ECSE 参数:")
    print(f"  Rank (r,q):  ({cfg.get('rank_evidence', 6)}, {cfg.get('rank_prior', 4)})")
    print(f"  Kappa:       {cfg.get('kappa', 0.6)}")
    print(f"  Lambda:      {cfg.get('lambda_prior', 0.3)}")
    print(f"  Anchor:      {cfg.get('anchor_layer', 26)}")
    print("=" * 80)
    print(f"生成参数:")
    print(f"  Temperature: {cfg.get('temperature', 0.2)}")
    print(f"  Top-P:       {cfg.get('top_p', 0.9)}")
    print(f"  Max Tokens:  {cfg.get('max_new_tokens', 128)}")
    print("=" * 80)
    
    # 构建数据集
    print(f"构建 CHAIR 数据集...")
    data = build_chair_dataset(
        split=split,
        data_root=data_root,
        sampling=sampling,
        num_samples=num_samples,
        seed=seed
    )
    print(f"加载 {len(data)} 张图像")
    
    # 初始化 ECSE 参数
    print("初始化 ECSE 参数...")
    ecse_cfg = ECSEConfig(
        rank_evidence=cfg.get("rank_evidence", 6),
        rank_prior=cfg.get("rank_prior", 4),
        kappa=cfg.get("kappa", 0.6),
        lambda_prior=cfg.get("lambda_prior", 0.3),
        eps=cfg.get("eps", 1e-6),
        # 稳健化参数（若配置未提供，则回退到 ECSEConfig 默认值）
        lambda_n_max=cfg.get("lambda_n_max", 4.0),
        lambda_p_max=cfg.get("lambda_p_max", 4.0),
        vcr_floor=cfg.get("vcr_floor", 0.05),
        pcr_ceiling=cfg.get("pcr_ceiling", 0.95),
        pcr_threshold=cfg.get("pcr_threshold", 0.02),
        blend_tau=cfg.get("blend_tau", 0.7),
        norm_preserve=cfg.get("norm_preserve", True),
        norm_beta=cfg.get("norm_beta", 0.5),
        weight_temp=cfg.get("weight_temp", 1.5),
        # 消融与变体
        uniform_svd=cfg.get("uniform_svd", False),
        no_complement=cfg.get("no_complement", False),
        no_gating=cfg.get("no_gating", False),
        use_fixed_strengths=cfg.get("use_fixed_strengths", False),
        fixed_lambda_n=cfg.get("fixed_lambda_n", 0.0),
        fixed_lambda_p=cfg.get("fixed_lambda_p", 0.0),
        only_residual=cfg.get("only_residual", False),
        only_anti_prior=cfg.get("only_anti_prior", False),
    )
    
    # 选择并初始化引擎
    engine_name = str(cfg.get("engine", "llava")).lower()
    print(f"初始化推理引擎: {engine_name}")
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and cfg.get("gpu_id") is not None:
        device = f"cuda:{cfg.get('gpu_id')}"
    
    if engine_name == "llava":
        # 处理 estimate_layer 和 edit_layer（支持 -1 表示顶层）
        estimate_layer = cfg.get("estimate_layer", None)
        edit_layer = cfg.get("edit_layer", None)
        if estimate_layer == -1:
            estimate_layer = -1  # 顶层
        if edit_layer == -1:
            edit_layer = -1  # 顶层
        
        eng_cfg = EngineConfig(
            model_name=cfg["model_name"],
            anchor_layer=cfg.get("anchor_layer", 26),
            estimate_layer=estimate_layer,
            edit_layer=edit_layer,
            visual_clean_layers=cfg.get("visual_clean_layers", [10]),
            multi_anchor_layers=cfg.get("multi_anchor_layers", None),
            layer_weighting=cfg.get("layer_weighting", "learned"),
            layer_weight_temp=cfg.get("layer_weight_temp", 1.0),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            precision=cfg.get("precision", "bf16")
        )
        engine = LLaVAECSEEngine(eng_cfg, ecse_cfg)
    elif engine_name == "minigpt4":
        from ecse.engines.minigpt4 import MiniGPT4ECSEEngine, MiniGPT4EngineConfig
        # 从 device 字符串中提取 gpu_id（如 "cuda:0" -> 0）
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                gpu_id = 0
        mg_cfg = MiniGPT4EngineConfig(
            cfg_path=cfg["minigpt4_cfg_path"],
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            gpu_id=cfg.get("gpu_id", gpu_id),
        )
        engine = MiniGPT4ECSEEngine(mg_cfg, ecse_cfg, device=device)
    elif engine_name in ["mplug", "mplug_owl2", "mplug-owl2"]:
        from ecse.engines.mplug_owl2 import MplugOwl2Engine, MplugOwl2EngineConfig
        if "model_path" not in cfg:
            raise ValueError("mPLUG-Owl2 需要配置 model_path（HF 或本地路径）")
        mplug_cfg = MplugOwl2EngineConfig(
            model_path=cfg["model_path"],
            model_name=cfg.get("model_name", "mplug_owl2"),
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            precision=cfg.get("precision", "fp16"),
        )
        engine = MplugOwl2Engine(mplug_cfg, ecse_cfg)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_name}")
    print("[ECSE] 引擎初始化完成")
    
    # 准备输出文件
    if args.output_file:
        output_file = args.output_file
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir,
            f"chair_captions_ecse_{split}_{timestamp}.jsonl"
        )
    
    print(f"输出文件: {output_file}")
    print("=" * 80)
    
    # 生成 caption
    # 根据引擎选择合适的提示模板
    if engine_name == "llava":
        prompt_template = "USER: <image>\n{question}\nASSISTANT:"
    elif engine_name in ["mplug", "mplug_owl2", "mplug-owl2"]:
        # mPLUG-Owl2 格式提示（与 POPE 评测保持一致）
        prompt_template = "USER: <|image|>\n{question}\nASSISTANT:"
    else:
        prompt_template = "{question}"
    written = 0
    total = len(data)
    
    with open(output_file, "w") as f:
        for idx, item in enumerate(tqdm(data, desc="生成 Caption"), start=1):
            image_id = int(item["image_id"])
            image_path = item["image_path"]
            question = item.get("question", "Please describe this image in detail.")
            
            if not os.path.exists(image_path):
                print(f"[WARN] 图像不存在: {image_path}")
                continue
            
            try:
                prompt = prompt_template.format(question=question)
                output = engine.generate(prompt, image_path)
                caption = output["text"]
                
                # 打印详细信息（可选）
                if args.verbose:
                    print(f"\n{'='*80}")
                    print(f"[进度: {idx}/{total}] [Image ID: {image_id}]")
                    print(f"[Question]: {question}")
                    print(f"[Caption]: {caption}")
                    
                    # 打印证据证书
                    if output["certs"]:
                        last_cert = output["certs"][-1]
                        print(f"[Certs] ECR={last_cert['ecr']:.4f}, "
                              f"EPC={last_cert['epc']:.4f}, "
                              f"Gate={last_cert['gate']:.4f}")
                    print(f"{'='*80}\n")
                    sys.stdout.flush()  # 立即刷新输出缓冲区
                
                # 写入 JSONL（与 Nullu 格式兼容）
                record = {
                    "image_id": image_id,
                    "caption": caption
                }
                f.write(json.dumps(record) + "\n")
                f.flush()  # 立即写入磁盘
                written += 1
                
            except Exception as e:
                print(f"[ERROR] 处理图像 {image_id} 失败: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                continue
    
    print(f"\n{'='*80}")
    print(f"完成！已生成 {written} 条 caption")
    print(f"保存至: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

