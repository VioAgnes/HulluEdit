"""
快速测试脚本（少样本验证）
用于验证 ECSE 系统是否正常工作
"""
import argparse
import json
import sys
from pathlib import Path
from glob import glob
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ecse.engines.llava7b import LLaVAECSEEngine, EngineConfig
from ecse.steer import ECSEConfig


def main():
    parser = argparse.ArgumentParser(description="ECSE 快速测试")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--num-images", type=int, default=10, help="测试图像数")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # 初始化引擎
    print("[快速测试] 初始化 ECSE 引擎...")
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
        max_new_tokens=50,  # 快速测试用较短长度
        top_p=cfg.get("top_p", 0.9),
        temperature=cfg.get("temperature", 0.2),
        precision=cfg.get("precision", "bf16")
    )
    
    engine = LLaVAECSEEngine(eng_cfg, ecse_cfg)
    
    # 加载测试图像
    coco_img_dir = Path(cfg["coco_images"])
    image_files = sorted(glob(str(coco_img_dir / "*.jpg")))[:args.num_images]
    
    if not image_files:
        print(f"[错误] 未找到图像: {coco_img_dir}")
        return
    
    print(f"[快速测试] 测试图像数: {len(image_files)}")
    
    # 测试生成
    test_prompts = [
        "USER: <image>\nWhat objects do you see in this image?\nASSISTANT:",
        "USER: <image>\nDescribe this image.\nASSISTANT:",
        "USER: <image>\nIs there a person in this image?\nASSISTANT:",
    ]
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"[测试 {i}/{len(image_files)}] {Path(img_path).name}")
        print(f"{'='*60}")
        
        prompt = test_prompts[i % len(test_prompts)]
        
        try:
            output = engine.generate(prompt, img_path, max_new_tokens=30)
            
            print(f"[输出] {output['text']}")
            
            if output["certs"]:
                last_cert = output["certs"][-1]
                print(f"[证书] ECR={last_cert['ecr']:.4f}, "
                      f"EPC={last_cert['epc']:.4f}, "
                      f"Gate={last_cert['gate']:.4f}")
            
        except Exception as e:
            print(f"[错误] {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("[快速测试] 完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

