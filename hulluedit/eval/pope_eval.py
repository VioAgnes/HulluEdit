"""
POPE (Polling-based Object Probing Evaluation) 评测
评估多模态模型的目标幻觉问题
支持 LLaVA-1.5、MiniGPT-4 和 mPLUG-Owl2
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import yaml
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ecse.engines.llava7b import LLaVAECSEEngine, EngineConfig
from ecse.engines.minigpt4 import MiniGPT4ECSEEngine, MiniGPT4EngineConfig
# mPlug-Owl2 延迟导入，只在需要时导入
from ecse.steer import ECSEConfig


def load_pope_data(pope_root: str, split: str = "adversarial"):
    """
    加载 POPE 数据集
    POPE 格式：每行一个 JSON，包含 image, text, label
    """
    pope_file = Path(pope_root) / f"coco_pope_{split}.json"
    if not pope_file.exists():
        raise FileNotFoundError(f"POPE file not found: {pope_file}")
    
    data = []
    with open(pope_file) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    return data


def evaluate_pope(predictions, labels):
    """计算 POPE 指标：Accuracy, Precision, Recall, F1"""
    assert len(predictions) == len(labels)
    
    tp = sum(1 for p, l in zip(predictions, labels) if p == "yes" and l == "yes")
    fp = sum(1 for p, l in zip(predictions, labels) if p == "yes" and l == "no")
    tn = sum(1 for p, l in zip(predictions, labels) if p == "no" and l == "no")
    fn = sum(1 for p, l in zip(predictions, labels) if p == "no" and l == "yes")
    
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def extract_yes_no(text: str) -> str:
    """从模型输出中提取 yes/no 答案"""
    if not text:
        return "no"
    
    text = text.lower().strip()
    
    # 检查前 30 个字符中是否包含 yes/no
    prefix = text[:30]
    
    # 优先检查 yes
    if "yes" in prefix:
        return "yes"
    elif "no" in prefix:
        return "no"
    else:
        # 默认返回 no（保守策略）
        return "no"


def main():
    parser = argparse.ArgumentParser(description="POPE 评测（ECSE，支持 LLaVA-1.5、MiniGPT-4 和 mPLUG-Owl2）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--split", type=str, default="adversarial", 
                       choices=["random", "popular", "adversarial"],
                       help="POPE 数据集分割")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大样本数（用于快速测试）")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 路径")
    parser.add_argument("--model-name", type=str, default=None,
                       help="模型名称（LLaVA、MiniGPT-4 或 mPLUG-Owl2），如果不指定则从配置文件推断")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径（mPLUG-Owl2 使用，如果提供则覆盖配置文件中的设置）")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 如果通过命令行提供了 model_path，则覆盖配置文件中的设置
    if args.model_path:
        cfg["model_path"] = args.model_path
    
    # 确定模型类型
    model_name = args.model_name or cfg.get("model_name", "")
    model_name_lower = model_name.lower()
    
    # 初始化 ECSE 配置
    ecse_cfg = ECSEConfig(
        rank_evidence=cfg.get("rank_evidence", 8),
        rank_prior=cfg.get("rank_prior", 5),
        kappa=cfg.get("kappa", 0.50),
        lambda_prior=cfg.get("lambda_prior", 0.22),
        eps=cfg.get("eps", 1e-6),
        lambda_n_max=cfg.get("lambda_n_max", 3.2),
        lambda_p_max=cfg.get("lambda_p_max", 3.8),
        vcr_floor=cfg.get("vcr_floor", 0.045),
        pcr_ceiling=cfg.get("pcr_ceiling", 0.92),
        pcr_threshold=cfg.get("pcr_threshold", 0.015),
        blend_tau=cfg.get("blend_tau", 0.76),
        norm_preserve=cfg.get("norm_preserve", True),
        norm_beta=cfg.get("norm_beta", 0.74),
        weight_temp=cfg.get("weight_temp", 1.15),
    )
    
    # 初始化引擎
    print(f"[POPE] 初始化 ECSE 引擎: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if "llava" in model_name_lower:
        eng_cfg = EngineConfig(
            model_name=cfg["model_name"],
            anchor_layer=cfg.get("anchor_layer", 28),
            visual_clean_layers=cfg.get("visual_clean_layers", [10]),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.12),
            precision=cfg.get("precision", "bf16")
        )
        engine = LLaVAECSEEngine(eng_cfg, ecse_cfg, device=device)
        is_llava = True
        is_mplug = False
    elif "minigpt" in model_name_lower:
        # 从 device 字符串中提取 gpu_id（如 "cuda:0" -> 0）
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                gpu_id = 0
        
        eng_cfg = MiniGPT4EngineConfig(
            cfg_path=cfg.get("minigpt4_cfg_path"),
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.12),
            gpu_id=cfg.get("gpu_id", gpu_id),
        )
        engine = MiniGPT4ECSEEngine(eng_cfg, ecse_cfg, device=device)
        is_llava = False
        is_mplug = False
    elif "mplug" in model_name_lower:
        # 延迟导入 mPlug-Owl2，只在需要时导入
        try:
            from ecse.engines.mplug_owl2 import MplugOwl2Engine, MplugOwl2EngineConfig
        except ImportError as e:
            raise ImportError(
                f"无法导入 mPlug-Owl2 模块。请确保已安装 mplug_owl2 依赖。"
                f"错误详情: {e}"
            )
        
        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("mPLUG-Owl2 需要配置 model_path（HF 或本地路径）")
        eng_cfg = MplugOwl2EngineConfig(
            model_path=model_path,
            anchor_layer=cfg.get("anchor_layer", 26),
            max_new_tokens=cfg.get("max_new_tokens", 128),
            top_p=cfg.get("top_p", 0.9),
            temperature=cfg.get("temperature", 0.12),
            precision=cfg.get("precision", "fp16")
        )
        engine = MplugOwl2Engine(eng_cfg, ecse_cfg, device=device)
        is_llava = False
        is_mplug = True
    else:
        raise ValueError(f"不支持的模型: {model_name}，请使用 LLaVA、MiniGPT-4 或 mPLUG-Owl2")
    
    # 加载 POPE 数据
    print(f"[POPE] 加载数据集: {args.split}")
    pope_data = load_pope_data(cfg["pope_root"], args.split)
    
    if args.max_samples:
        pope_data = pope_data[:args.max_samples]
    
    print(f"[POPE] 样本数: {len(pope_data)}")
    
    # 推理
    results = []
    predictions = []
    labels = []
    
    coco_img_dir = Path(cfg["coco_images"])
    
    # 打印频率：每 N 个样本打印一次详细信息，或者每个样本都打印（如果样本数较少）
    print_every = max(1, len(pope_data) // 100) if len(pope_data) > 100 else 1
    
    for idx, item in enumerate(tqdm(pope_data, desc="POPE 评测")):
        # item["image"] 可能是相对路径或绝对路径
        if "/" in item["image"] and Path(item["image"]).exists():
            image_path = Path(item["image"])
        else:
            image_path = coco_img_dir / item["image"]
        
        if not image_path.exists():
            tqdm.write(f"[WARNING] 图像不存在: {image_path}")
            continue
        
        question = item["text"]
        label = item["label"]  # "yes" or "no"
        
        # 根据模型类型构建不同的提示
        if is_llava:
            # LLaVA 格式提示
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
        elif is_mplug:
            # mPLUG-Owl2 格式提示
            prompt = f"USER: <|image|>\n{question}\nASSISTANT:"
        else:
            # MiniGPT-4 格式提示（直接使用问题）
            prompt = question
        
        # 生成
        try:
            if is_llava:
                # LLaVA 接受 PIL Image 或路径
                image = Image.open(image_path).convert("RGB")
                output_dict = engine.generate(prompt, image, max_new_tokens=cfg.get("max_new_tokens", 128))
                pred_text = output_dict.get("text", output_dict.get("output", ""))
            elif is_mplug:
                # mPLUG-Owl2 接受路径
                output_dict = engine.generate(prompt, str(image_path), max_new_tokens=cfg.get("max_new_tokens", 128))
                pred_text = output_dict.get("text", output_dict.get("output", ""))
            else:
                # MiniGPT-4 接受路径
                output_dict = engine.generate(prompt, str(image_path), max_new_tokens=cfg.get("max_new_tokens", 128))
                pred_text = output_dict.get("text", output_dict.get("output", ""))
            
            pred_label = extract_yes_no(pred_text)
            
            # 计算 ECSE 指标的平均值（如果有）
            certs = output_dict.get("certs", [])
            avg_ecr = sum(c.get("ecr", 0.0) for c in certs) / len(certs) if certs else 0.0
            avg_epc = sum(c.get("epc", 0.0) for c in certs) / len(certs) if certs else 0.0
            avg_gate = sum(c.get("gate", 0.0) for c in certs) / len(certs) if certs else 0.0
            
            # 判断是否正确
            is_correct = "✓" if pred_label == label else "✗"
            
            # 打印输出（使用 tqdm.write 避免干扰进度条）
            if idx % print_every == 0 or idx < 10:  # 前10个样本总是打印
                tqdm.write(f"\n[样本 {idx+1}/{len(pope_data)}]")
                tqdm.write(f"  问题: {question}")
                tqdm.write(f"  生成: {pred_text[:200]}{'...' if len(pred_text) > 200 else ''}")
                tqdm.write(f"  提取: {pred_label} | 标签: {label} {is_correct}")
                if certs:
                    tqdm.write(f"  ECSE: ECR={avg_ecr:.3f}, EPC={avg_epc:.3f}, Gate={avg_gate:.3f}")
            
            results.append({
                "image": item["image"],
                "question": question,
                "label": label,
                "prediction": pred_label,
                "raw_output": pred_text,
                "certs": certs
            })
            
            predictions.append(pred_label)
            labels.append(label)
            
            # 定期打印中间统计信息（每100个样本）
            if len(predictions) > 0 and len(predictions) % 100 == 0:
                temp_metrics = evaluate_pope(predictions, labels)
                tqdm.write(f"\n[中间统计] 已处理 {len(predictions)} 个样本")
                tqdm.write(f"  当前准确率: {temp_metrics['accuracy']:.4f}")
                tqdm.write(f"  当前 F1: {temp_metrics['f1']:.4f}")
                tqdm.write(f"  TP/FP/TN/FN: {temp_metrics['tp']}/{temp_metrics['fp']}/{temp_metrics['tn']}/{temp_metrics['fn']}\n")
            
        except Exception as e:
            tqdm.write(f"[ERROR] 样本 {idx+1} ({image_path}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 计算指标
    if len(predictions) == 0:
        print("[ERROR] 没有成功生成任何预测结果")
        return
    
    metrics = evaluate_pope(predictions, labels)
    
    print("\n[POPE 结果]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}")
    
    # 保存结果
    output_data = {
        "config": args.config,
        "model_name": model_name,
        "split": args.split,
        "num_samples": len(results),
        "metrics": metrics,
        "results": results
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[POPE] 结果已保存: {args.output}")


if __name__ == "__main__":
    main()

