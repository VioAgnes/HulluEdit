"""
COCO Caption 生成评测（用于 CHAIR 计算）
"""
import argparse
import json
import os
import sys
import random
from pathlib import Path
from glob import glob
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ecse.engines.llava7b import LLaVAECSEEngine, EngineConfig
from ecse.steer import ECSEConfig


def main():
    parser = argparse.ArgumentParser(description="COCO Caption 生成（ECSE）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大样本数（用于快速测试）")
    parser.add_argument("--output", type=str, required=True, help="输出 JSON 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 初始化引擎
    ecse_cfg = ECSEConfig(
        rank_evidence=cfg.get("rank_evidence", 6),
        rank_prior=cfg.get("rank_prior", 4),
        kappa=cfg.get("kappa", 0.6),
        lambda_prior=cfg.get("lambda_prior", 0.3),
        eps=cfg.get("eps", 1e-6)
    )
    
    eng_cfg = EngineConfig(
        model_name=cfg["model_name"],
        anchor_layer=cfg.get("anchor_layer", 26),
        visual_clean_layers=cfg.get("visual_clean_layers", [10]),
        max_new_tokens=cfg.get("max_new_tokens", 128),
        top_p=cfg.get("top_p", 0.9),
        temperature=cfg.get("temperature", 0.2),
        precision=cfg.get("precision", "bf16")
    )
    
    print("[Caption] 初始化 ECSE 引擎...")
    engine = LLaVAECSEEngine(eng_cfg, ecse_cfg)
    
    # 加载图像列表
    coco_img_dir = Path(cfg["coco_images"])
    image_files = sorted(glob(str(coco_img_dir / "*.jpg")))
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {coco_img_dir}")
    
    # 随机采样
    random.shuffle(image_files)
    if args.max_samples:
        image_files = image_files[:args.max_samples]
    
    print(f"[Caption] 图像数: {len(image_files)}")
    
    # 生成 caption
    results = []
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    
    for img_path in tqdm(image_files, desc="生成 Caption"):
        image_id = Path(img_path).stem  # COCO_val2014_000000123456
        
        try:
            output = engine.generate(prompt, img_path)
            
            results.append({
                "image_id": image_id,
                "image_path": str(img_path),
                "caption": output["text"],
                "certs": output["certs"]
            })
            
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            continue
    
    # 保存结果
    output_data = {
        "config": args.config,
        "num_samples": len(results),
        "results": results
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[Caption] 结果已保存: {args.output}")
    print(f"  样本数: {len(results)}")
    
    # 显示统计
    if results:
        avg_ecr = sum(r["certs"][-1]["ecr"] for r in results if r["certs"]) / len(results)
        avg_epc = sum(r["certs"][-1]["epc"] for r in results if r["certs"]) / len(results)
        print(f"  平均 ECR: {avg_ecr:.4f}")
        print(f"  平均 EPC: {avg_epc:.4f}")


if __name__ == "__main__":
    main()

