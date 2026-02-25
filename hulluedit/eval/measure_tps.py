#!/usr/bin/env python3
"""
TPS (Tokens Per Second) 测量脚本
用于对比不同方法的解码效率

测量逻辑：
1. 仅统计解码阶段的耗时（不包括图像预处理、模型加载）
2. 统计实际生成的 token 数量
3. 计算 TPS = 总生成token数 / 总解码耗时
4. 支持多次运行取平均值（默认 5 次）
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import yaml
import random
import numpy as np
import torch
from typing import Dict, List, Optional, Any

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
    parser = argparse.ArgumentParser(description="测量 TPS (Tokens Per Second)")
    
    # 配置文件
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    
    # 数据集参数
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--num-samples", type=int, default=500, help="测试样本数")
    parser.add_argument("--sampling", type=str, default="random", choices=["first", "random"])
    
    # TPS 测量参数
    parser.add_argument("--num-runs", type=int, default=5, help="重复运行次数（用于取平均）")
    parser.add_argument("--warmup", type=int, default=3, help="预热轮数（不计入统计）")
    
    # 输出
    parser.add_argument("--output-file", type=str, default=None, help="输出 JSON 文件路径")
    
    # 其他
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")
    
    return parser.parse_args()


class TPSTimer:
    """TPS 计时器，仅统计解码阶段"""
    
    def __init__(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.num_samples = 0
        
    def start(self):
        """开始计时（应在解码开始前调用）"""
        self._start_time = time.time()
        
    def end(self, num_tokens: int):
        """结束计时并记录 token 数（应在解码结束后调用）"""
        elapsed = time.time() - self._start_time
        self.total_time += elapsed
        self.total_tokens += num_tokens
        self.num_samples += 1
        
    def get_tps(self) -> float:
        """计算平均 TPS"""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time
    
    def reset(self):
        """重置统计"""
        self.total_tokens = 0
        self.total_time = 0.0
        self.num_samples = 0


def measure_tps_for_method(
    engine,
    data: List[Dict],
    prompt_template: str,
    max_new_tokens: int,
    warmup: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    测量单个方法的 TPS
    
    Returns:
        {
            "tps": float,
            "total_tokens": int,
            "total_time": float,
            "num_samples": int,
            "avg_time_per_sample": float,
            "avg_tokens_per_sample": float
        }
    """
    timer = TPSTimer()
    
    # 预热（不计入统计）
    if warmup > 0:
        warmup_data = data[:min(warmup, len(data))]
        for item in warmup_data:
            try:
                image_path = item["image_path"]
                question = item.get("question", "Please describe this image in detail.")
                prompt = prompt_template.format(question=question)
                engine.generate(prompt, image_path)
            except Exception as e:
                if verbose:
                    print(f"[WARN] 预热失败: {e}")
    
    # 正式测量
    for item in tqdm(data[warmup:], desc="测量 TPS", disable=not verbose):
        image_path = item["image_path"]
        question = item.get("question", "Please describe this image in detail.")
        
        if not os.path.exists(image_path):
            continue
        
        try:
            prompt = prompt_template.format(question=question)
            
            # 开始计时（解码阶段）
            timer.start()
            
            # 生成（这里假设 engine.generate 返回包含 token 数的信息）
            output = engine.generate(prompt, image_path, max_new_tokens=max_new_tokens)
            
            # 计算生成的 token 数
            # 优先使用 tokens 列表的长度（最准确）
            if "tokens" in output and isinstance(output["tokens"], list):
                num_tokens = len(output["tokens"])
            elif "num_tokens" in output:
                num_tokens = output["num_tokens"]
            elif "text" in output:
                # 从文本估算（简单方法：按空格分割，不够准确但作为备选）
                num_tokens = len(output["text"].split())
            else:
                # 默认值（应该不会到这里）
                num_tokens = max_new_tokens
            
            # 结束计时
            timer.end(num_tokens)
            
        except Exception as e:
            if verbose:
                print(f"[ERROR] 处理失败: {e}")
            continue
    
    return {
        "tps": timer.get_tps(),
        "total_tokens": timer.total_tokens,
        "total_time": timer.total_time,
        "num_samples": timer.num_samples,
        "avg_time_per_sample": timer.total_time / max(timer.num_samples, 1),
        "avg_tokens_per_sample": timer.total_tokens / max(timer.num_samples, 1)
    }


def main():
    args = parse_args()
    
    # 加载配置文件
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 设置随机种子
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    
    # 数据集参数
    split = args.split
    sampling = args.sampling
    num_samples = args.num_samples
    data_root = cfg.get("coco_root", "/data/home/scyb531/DATA/")
    
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
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and cfg.get("gpu_id") is not None:
        device = f"cuda:{cfg.get('gpu_id')}"
    
    # 初始化 ECSE 参数
    ecse_cfg = ECSEConfig(
        rank_evidence=cfg.get("rank_evidence", 6),
        rank_prior=cfg.get("rank_prior", 4),
        kappa=cfg.get("kappa", 0.6),
        lambda_prior=cfg.get("lambda_prior", 0.3),
        eps=cfg.get("eps", 1e-6),
        lambda_n_max=cfg.get("lambda_n_max", 4.0),
        lambda_p_max=cfg.get("lambda_p_max", 4.0),
        vcr_floor=cfg.get("vcr_floor", 0.05),
        pcr_ceiling=cfg.get("pcr_ceiling", 0.95),
        pcr_threshold=cfg.get("pcr_threshold", 0.02),
        blend_tau=cfg.get("blend_tau", 0.7),
        norm_preserve=cfg.get("norm_preserve", True),
        norm_beta=cfg.get("norm_beta", 0.5),
        weight_temp=cfg.get("weight_temp", 1.5),
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
    max_new_tokens = cfg.get("max_new_tokens", 64)
    
    print(f"初始化推理引擎: {engine_name}")
    
    if engine_name == "llava":
        estimate_layer = cfg.get("estimate_layer", None)
        edit_layer = cfg.get("edit_layer", None)
        if estimate_layer == -1:
            estimate_layer = -1
        if edit_layer == -1:
            edit_layer = -1
        
        eng_cfg = EngineConfig(
            model_name=cfg["model_name"],
            anchor_layer=cfg.get("anchor_layer", 26),
            estimate_layer=estimate_layer,
            edit_layer=edit_layer,
            visual_clean_layers=cfg.get("visual_clean_layers", [10]),
            multi_anchor_layers=cfg.get("multi_anchor_layers", None),
            layer_weighting=cfg.get("layer_weighting", "learned"),
            layer_weight_temp=cfg.get("layer_weight_temp", 1.0),
            max_new_tokens=max_new_tokens,
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            precision=cfg.get("precision", "bf16")
        )
        engine = LLaVAECSEEngine(eng_cfg, ecse_cfg)
        prompt_template = "USER: <image>\n{question}\nASSISTANT:"
    elif engine_name == "minigpt4":
        from ecse.engines.minigpt4 import MiniGPT4ECSEEngine, MiniGPT4EngineConfig
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                gpu_id = 0
        mg_cfg = MiniGPT4EngineConfig(
            cfg_path=cfg["minigpt4_cfg_path"],
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=max_new_tokens,
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            gpu_id=cfg.get("gpu_id", gpu_id),
        )
        engine = MiniGPT4ECSEEngine(mg_cfg, ecse_cfg, device=device)
        prompt_template = "{question}"
    elif engine_name in ["mplug", "mplug_owl2", "mplug-owl2"]:
        from ecse.engines.mplug_owl2 import MplugOwl2Engine, MplugOwl2EngineConfig
        if "model_path" not in cfg:
            raise ValueError("mPLUG-Owl2 需要配置 model_path")
        mplug_cfg = MplugOwl2EngineConfig(
            model_path=cfg["model_path"],
            model_name=cfg.get("model_name", "mplug_owl2"),
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=max_new_tokens,
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.2),
            precision=cfg.get("precision", "fp16"),
        )
        engine = MplugOwl2Engine(mplug_cfg, ecse_cfg)
        prompt_template = "USER: <|image|>\n{question}\nASSISTANT:"
    else:
        raise ValueError(f"不支持的引擎类型: {engine_name}")
    
    print("[ECSE] 引擎初始化完成")
    print("=" * 80)
    
    # 多次运行测量 TPS
    all_results = []
    for run_idx in range(args.num_runs):
        print(f"\n运行 {run_idx + 1}/{args.num_runs}...")
        setup_seeds(seed + run_idx)  # 每次运行使用不同的种子
        
        result = measure_tps_for_method(
            engine=engine,
            data=data,
            prompt_template=prompt_template,
            max_new_tokens=max_new_tokens,
            warmup=args.warmup,
            verbose=args.verbose
        )
        all_results.append(result)
        
        if args.verbose:
            print(f"  TPS: {result['tps']:.2f} tokens/s")
            print(f"  总 tokens: {result['total_tokens']}")
            print(f"  总时间: {result['total_time']:.2f}s")
    
    # 计算平均值和标准差
    tps_values = [r["tps"] for r in all_results]
    avg_tps = np.mean(tps_values)
    std_tps = np.std(tps_values)
    
    # 汇总结果
    summary = {
        "method": cfg.get("method_name", "HulluEdit"),
        "engine": engine_name,
        "num_samples": len(data),
        "num_runs": args.num_runs,
        "max_new_tokens": max_new_tokens,
        "tps_mean": float(avg_tps),
        "tps_std": float(std_tps),
        "tps_min": float(np.min(tps_values)),
        "tps_max": float(np.max(tps_values)),
        "runs": all_results
    }
    
    # 输出结果
    print("\n" + "=" * 80)
    print("TPS 测量结果")
    print("=" * 80)
    print(f"方法:        {summary['method']}")
    print(f"引擎:        {summary['engine']}")
    print(f"样本数:      {summary['num_samples']}")
    print(f"运行次数:    {summary['num_runs']}")
    print(f"最大 tokens: {summary['max_new_tokens']}")
    print(f"平均 TPS:    {avg_tps:.2f} ± {std_tps:.2f} tokens/s")
    print(f"TPS 范围:    [{np.min(tps_values):.2f}, {np.max(tps_values):.2f}] tokens/s")
    print("=" * 80)
    
    # 保存结果
    if args.output_file:
        output_file = args.output_file
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = cfg.get("output_dir", "/data/home/scyb531/lyg/HulluEdit/outputs/tps")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"tps_{engine_name}_{summary['method'].lower().replace(' ', '_')}_{timestamp}.json"
        )
    
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存至: {output_file}")
    
    return summary


if __name__ == "__main__":
    main()

